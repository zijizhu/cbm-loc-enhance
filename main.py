from typing import Iterator
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

from data import load_data
from nets import PPConceptNet, Criterion
from lightning import seed_everything
from torchmetrics.classification import BinaryAccuracy

from sem_analysis.vlpart import generate_prototype_semantics
from eval.concept_locality import Cub2011Eval, evaluate_concept_locality, get_activation_maps


def train(
  model: nn.Module,
  train_loader: Iterator,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.dtype | str,
  with_concepts: bool = False,
):
  model.train()
  train_losses = defaultdict(float)
  correct = 0
  total = 0
  bin_acc = BinaryAccuracy(multidim_average="global").to(device=device)

  for images, labels, attributes in tqdm(train_loader):
    images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)

    logits, concept_scores, cosine_scores, cosine_activations, activations = model(images, with_concepts=with_concepts)
    loss, loss_dict = criterion(logits, concept_scores, cosine_scores, labels, attributes, model.prototypes)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if not with_concepts:
      model.normalize_prototypes()

    for loss_name, loss_value in loss_dict.items():
      train_losses[loss_name] += loss_dict[loss_name].item()

    predicted = torch.argmax(logits, dim=-1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    if with_concepts:
      bin_acc(torch.sigmoid(concept_scores), attributes)

  for loss_name, loss_value in train_losses.items():
    train_losses[loss_name] = loss_value / len(train_loader)
  return train_losses, correct / total, bin_acc.compute().item() if with_concepts else None


def validate(model: nn.Module, test_loader: Iterator, criterion: nn.Module, device: torch.dtype | str, with_concepts: bool = False):
  model.eval()
  val_losses = defaultdict(float)
  correct = 0
  total = 0
  bin_acc = BinaryAccuracy(multidim_average="global").to(device=device)

  with torch.no_grad():
    for images, labels, attributes in tqdm(test_loader):
      images, labels, attributes = images.to(device), labels.to(device), attributes.to(device)

      logits, concept_scores, cosine_scores, cosine_activations, activations = model(images, with_concepts=with_concepts)
      loss, loss_dict = criterion(logits, concept_scores, cosine_scores, labels, attributes, model.prototypes)

      for loss_name, loss_value in loss_dict.items():
        val_losses[loss_name] += loss_dict[loss_name].item()

      predicted = torch.argmax(logits, dim=-1)
      correct += (predicted == labels).sum().item()
      total += labels.size(0)
    if with_concepts:
      bin_acc(torch.sigmoid(concept_scores), attributes)

  for loss_name, loss_value in val_losses.items():
    val_losses[loss_name] = loss_value / len(test_loader)
  return val_losses, correct / total, bin_acc.compute().item(), bin_acc.compute().item() if with_concepts else None


def get_warmup_optimizer(model: nn.Module):
  optimizer = optim.Adam([
    {"params": model.adapter.parameters(), "lr": 3e-3, "weight_decay": 1e-3},
    {"params": [model.prototypes], "lr": 3e-3},
    {"params": model.score_aggregation.parameters(), "lr": 1e-06},
  ])

  for params in model.backbone.parameters():
    params.requires_grad = False

  return optimizer


def get_full_optimizer(model: nn.Module):
  optimizer = optim.Adam([
    {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 1e-3},
    {"params": model.adapter.parameters(), "lr": 3e-3, "weight_decay": 1e-3},
    {"params": [model.prototypes], "lr": 3e-3},
    {"params": model.score_aggregation.parameters(), "lr": 1e-06},
  ])

  for params in model.parameters():
    params.requires_grad = True

  return optimizer


def get_concept_layer_optimizer(model: nn.Module):
  """Tweak this function to set the try out different hyperparameters for training concept layer"""
  optimizer = optim.Adam([
      {'params': model.prototype_to_concept, 'lr': 1e-3, 'weight_decay': 1e-3},
      {'params': model.p2c_mask, 'lr': 1e-6},
      {'params': model.concept_to_class.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
  ])

  for params in model.parameters():
    params.requires_grad = False

  # Fine-tine concept layer only
  model.prototype_to_concept.requires_grad = True
  model.p2c_mask.requires_grad = True

  for params in model.concept_to_class.parameters():
      params.requires_grad = True

  return optimizer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, required=True)
  parser.add_argument("--evaluate", action="store_true")

  parser.add_argument("--backbone", type=str, default="densenet161", choices=["densenet161", "densenet121", "resnet34", "resnet18"])

  parser.add_argument("--batch-size", type=int, default=80, help="Batch size for training")  # 120 should word the best with CUB-200-2011
  parser.add_argument("--data-dir", type=str, default="datasets")
  parser.add_argument("--log-dir", type=str, default="logs")
  parser.add_argument("--pkl-dataset", action="store_true")

  parser.add_argument("--dataset", type=str, default="CUB", choices=["CUB", "SUN", "CelebA"])

  parser.add_argument("--k", type=int, default=10)

  parser.add_argument("--clst-coef", type=float, default=-0.8)
  parser.add_argument("--sep-coef", type=float, default=0.08)
  parser.add_argument("--bce-coef", type=float, default=1.0)
  parser.add_argument("--ortho-coef", type=float, default=1e-4)
  parser.add_argument("--disable-mask", action="store_true")
  parser.add_argument("--disable-basis-projection", action="store_true")

  parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
  parser.add_argument("--epochs", type=int, default=11, help="Number of training epochs")
  parser.add_argument("--joint-start-epoch", type=int, default=3)
  parser.add_argument("--concept-layer-start-epoch", type=int, default=6)

  parser.add_argument("--concept-layer-only", action="store_true")
  parser.add_argument("--resume", type=str)

  parser.add_argument("--seed", type=int, default=43)

  args = parser.parse_args()

  seed_everything(args.seed)
  torch.Generator().manual_seed(args.seed)

  log_dir = Path(args.log_dir) / args.name
  log_dir.mkdir(parents=True, exist_ok=True)

  logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
      logging.FileHandler((log_dir / "train.log").as_posix()),
      logging.StreamHandler(sys.stdout),
    ],
    force=True,
  )

  logger = logging.getLogger()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  logger.info(f"Training on {str(device)}")

  train_loader, test_loader, inference_loader, num_classes, num_concepts = load_data(
    args.dataset,
    args.data_dir,
    args.batch_size,
    seed=args.seed,
    pkl_dataset=args.pkl_dataset
  )

  concept_loc_dataset_eval, concept_loc_dataloader_eval = None, None
  if args.evaluate:
    # fmt: off
    transforms = Compose([
        Resize((224,224,)),
        ToTensor(),
        Normalize((0.485,0.456,0.406,),(0.229,0.224,0.225,),),
    ])
    concept_loc_dataset_eval = Cub2011Eval(root=args.data_dir, train=False, transform=transforms)
    concept_loc_dataloader_eval = DataLoader(concept_loc_dataset_eval, shuffle=False, batch_size=150)

  model = PPConceptNet(
    backbone_name=args.backbone,
    num_classes=num_classes,
    k=args.k,
    num_concepts=num_concepts,
    use_mask=not args.disable_mask,
    use_basis_projection=not args.disable_basis_projection,
  )
  if args.resume:
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt["state_dict"])

  criterion = Criterion(
    clst_coef=args.clst_coef, sep_coef=args.sep_coef, ortho_coef=args.ortho_coef, bce_coef=args.bce_coef, k=args.k, num_classes=num_classes
  )

  optimizer = get_warmup_optimizer(model)
  lr_scheduler = None

  model.to(device=device)
  criterion.to(device=device)

  best_val_acc = 0.0

  logger.info("Start warmup...")
  for epoch in range(args.epochs):
    logger.info(f"------------------------ Epoch {epoch} ------------------------")
    start_training_concept_layer = (epoch == 0) if args.concept_layer_only else (epoch == args.concept_layer_start_epoch)
    if start_training_concept_layer:
      logger.info("Start generating prototype semantics...")
      prototype_concept_mask = generate_prototype_semantics(
        model, inference_loader, log_dir, k=args.k, dataset_name=args.dataset, num_classes=num_classes, device=str(device)
      )

      # logger.warning("Prototype semantics generated as full of ones...")
      # prototype_concept_mask = torch.ones(
      #   (
      #     num_classes * args.k,
      #     num_concepts,
      #   ),
      #   dtype=torch.bool,
      # )

      model.init_concept_layer(prototype_concept_mask.float().to(device=device))

      logger.info("Start training concept layer...")
      optimizer = get_concept_layer_optimizer(model)
      lr_scheduler = None
    elif (not args.concept_layer_only) and (epoch == args.joint_start_epoch):
      logger.info("Start fine-tuning...")
      optimizer = get_full_optimizer(model)
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    with_concepts = args.concept_layer_only or epoch >= args.concept_layer_start_epoch
    logger.info(f"Training with concepts: {with_concepts}")

    train_losses, train_acc, train_cpt_acc = train(model, train_loader, criterion, optimizer, device, with_concepts=with_concepts)
    val_losses, val_acc, val_cpt_acc = validate(model, test_loader, criterion, device, with_concepts=with_concepts)

    for loss_name, loss_value in train_losses.items():
      logger.info(f"Train {loss_name}: {loss_value:.4f}")
    logger.info(f"Train Acc: {train_acc:.4f}")
    if with_concepts:
      logger.info(f"Train Concept Acc: {train_cpt_acc:.4f}")

    for loss_name, loss_value in val_losses.items():
      logger.info(f"Val {loss_name}: {loss_value:.4f}")
    logger.info(f"Val Acc: {val_acc:.4f}")
    if with_concepts:
      logger.info(f"Val Concept Acc: {val_cpt_acc:.4f}")

    # Checkpointing
    if val_acc > best_val_acc:
      torch.save(
        dict(
          state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
          hparams=vars(args),
        ),
        f"logs/{args.name}/model_best.pth",
      )
      logger.info("Model saved as model_best.pth")
      best_val_acc = val_acc

    save_current_model = args.concept_layer_only or (epoch >= args.concept_layer_start_epoch)
    if save_current_model:
      torch.save(
        dict(
          state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
          hparams=vars(args),
        ),
        f"logs/{args.name}/model_epoch{epoch}.pth",
      )
      logger.info(f"Model saved as model_epoch{epoch}.pth")

    # Evaluate concept trustworthiness
    if args.evaluate:
      model.eval()
      logger.info("Evaluating concept trustworthiness...")
      model.attributes_predictor = model.prototype_to_concept.T * F.relu(model.p2c_mask).T
      model.eval()
      all_activation_maps, all_img_ids = get_activation_maps(model, concept_loc_dataloader_eval)
      mean_loc_acc, _ = evaluate_concept_locality(all_activation_maps, all_img_ids, bbox_half_size=45)
      logger.info(f"Concept trustworthiness score of the network on the {len(concept_loc_dataset_eval)} test images: {mean_loc_acc:.2f}%")

    if lr_scheduler is not None:
      lr_scheduler.step()


if __name__ == "__main__":
  main()
