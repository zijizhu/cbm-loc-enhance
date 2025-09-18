import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from torchvision.models import (
  densenet161,
  DenseNet161_Weights,
  densenet121,
  DenseNet121_Weights,
  resnet18,
  ResNet18_Weights,
  resnet34,
  ResNet34_Weights,
)


class ScoreAggregation(nn.Module):
  def __init__(self, init_val: float = 0.2, num_classes: int = 200, k: int = 10) -> None:
    super().__init__()
    self.weights = nn.Parameter(torch.ones(num_classes, k).float())
    self.num_classes = num_classes

  def forward(self, x: torch.Tensor):
    n_classes, n_prototypes = self.weights.shape
    batch_size = x.size(0)
    sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes

    x = x.reshape(batch_size, self.num_classes, -1)
    x = x * sa_weights
    x = x.sum(-1)
    return x


class PPNet(nn.Module):
  def __init__(self, num_classes: int = 200, k: int = 10, dim: int = 64, score_aggregation: bool = True):
    super().__init__()
    backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
    self.backbone = nn.Sequential(*list(backbone.children())[:-1])
    self.k = k
    self.num_classes = num_classes
    self.dim = dim

    self.adapter = nn.Sequential(nn.Conv2d(in_channels=backbone.classifier.in_features, out_channels=dim, kernel_size=1), nn.Sigmoid())
    self.prototypes = nn.Parameter(torch.rand(num_classes * k, dim, 1, 1))

    if score_aggregation:
      self.classifier = ScoreAggregation(num_classes=num_classes, k=k)
    else:
      self.classifier = nn.Linear(num_classes * k, num_classes, bias=False)

    for layer in self.adapter.modules():
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        if layer.bias is not None:
          nn.init.constant_(layer.bias, 0)

  def forward(self, images: torch.Tensor):
    features = self.backbone(images)  # shape: [batch_size, dim, w, h]
    features = self.adapter(features)

    cosine_activations = cosine_conv2d(features, self.prototypes)
    activations = project2basis(features, self.prototypes)

    cosine_scores = F.adaptive_max_pool2d(cosine_activations, [1, 1]).squeeze()  # shape: [batch_size, num_concept * k]
    prototype_logits = F.adaptive_max_pool2d(activations, [1, 1]).squeeze()  # shape: [batch_size, num_concept * k]

    logits = self.classifier(prototype_logits)

    return logits, cosine_scores, cosine_activations, activations

  def normalize_prototypes(self):
    self.prototypes.data = F.normalize(self.prototypes, p=2, dim=1).data


def get_backbone(name: str) -> tuple[nn.Module, int]:
  assert name in ["densenet161", "densenet121", "resnet34", "resnet18"]
  if name == "densenet161":
    backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
    return nn.Sequential(*list(backbone.children())[:-1]), backbone.classifier.in_features
  elif name == "densenet121":
    backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
    return nn.Sequential(*list(backbone.children())[:-1]), backbone.classifier.in_features
  elif name == "resnet18":
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    return nn.Sequential(*list(backbone.children())[:-2]), backbone.fc.in_features
  elif name == "resnet34":
    backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
    return nn.Sequential(*list(backbone.children())[:-2]), backbone.fc.in_features


class PPConceptNet(nn.Module):
  def __init__(
    self,
    backbone_name: str = "densenet161",
    num_classes: int = 200,
    num_concepts: int = 112,
    k: int = 10,
    dim: int = 64,
    use_mask: bool = False,
    use_basis_projection=True,
  ):
    super().__init__()
    self.backbone, backbone_dim = get_backbone(backbone_name)
    self.k = k
    self.num_classes = num_classes
    self.dim = dim
    self.use_basis_projection = use_basis_projection

    self.adapter = nn.Sequential(nn.Conv2d(in_channels=backbone_dim, out_channels=dim, kernel_size=1), nn.Sigmoid())
    self.prototypes = nn.Parameter(torch.empty(num_classes * k, dim, 1, 1))

    self.score_aggregation = ScoreAggregation(num_classes=num_classes, k=k)

    self.prototype_to_concept = nn.Parameter(torch.zeros(num_classes * k, num_concepts))
    self.p2c_mask = nn.Parameter(torch.empty(num_classes * k, num_concepts)) if use_mask else None
    self.concept_to_class = nn.Linear(num_concepts, num_classes, bias=False)

    nn.init.kaiming_normal_(self.prototypes, a=sqrt(5))
    nn.init.kaiming_normal_(self.adapter[0].weight, mode="fan_out", nonlinearity="relu")
    nn.init.constant_(self.adapter[0].bias, 0)

  def init_concept_layer(self, mask: torch.Tensor):
    self.p2c_mask.data = mask
    nn.init.kaiming_normal_(self.prototype_to_concept, a=sqrt(5))
    pass

  def forward(self, images: torch.Tensor, with_concepts: bool = False):
    features = self.backbone(images)  # shape: [batch_size, dim, w, h]
    features = self.adapter(features)

    if with_concepts:
      activations = F.conv2d(features, self.prototypes)
      cosine_activations, cosine_scores = None, None
      prototype_logits = F.adaptive_max_pool2d(activations, [1, 1]).squeeze()

      if self.p2c_mask is not None:
        concept_scores = prototype_logits @ (self.prototype_to_concept * F.relu(self.p2c_mask))
      else:
        concept_scores = prototype_logits @ self.prototype_to_concept
      logits = self.concept_to_class(F.sigmoid(concept_scores))
    else:
      activations = project2basis(features, self.prototypes) if self.use_basis_projection else F.conv2d(features, self.prototypes)
      cosine_activations = cosine_conv2d(features, self.prototypes)
      cosine_scores = F.adaptive_max_pool2d(cosine_activations, [1, 1]).squeeze()  # shape: [batch_size, num_concept * k]
      concept_scores = None
      prototype_logits = F.adaptive_max_pool2d(activations, [1, 1]).squeeze()  # shape: [batch_size, num_concept * k]
      logits = self.score_aggregation(prototype_logits)

    return logits, concept_scores, cosine_scores, cosine_activations, activations

  @torch.no_grad()
  def push_forward(self, images: torch.Tensor):
      features = self.backbone(images)  # shape: [batch_size, dim, w, h]
      features = self.adapter(features)

      activations = project2basis(features, self.prototypes)
      return None, activations


  def normalize_prototypes(self):
    self.prototypes.data = F.normalize(self.prototypes, p=2, dim=1).data


def cosine_conv2d(x: torch.Tensor, weight: torch.Tensor):
  x = F.normalize(x, p=2, dim=1)
  weight = F.normalize(weight, p=2, dim=1)
  return F.conv2d(input=x, weight=weight)


def project2basis(x: torch.Tensor, weight: torch.Tensor):
  weight = F.normalize(weight, p=2, dim=1)
  return F.conv2d(input=x, weight=weight)


class Criterion(nn.Module):
  def __init__(self, clst_coef: float, bce_coef: float, sep_coef: float, ortho_coef: float, k: int = 10, num_classes: int = 200):
    super().__init__()
    self.num_classes = num_classes
    self.k = k
    self.xe = nn.CrossEntropyLoss()

    self.clst_coef = clst_coef
    self.sep_coef = sep_coef
    self.ortho_coef = ortho_coef
    self.bce_coef = bce_coef
    self.bce = nn.BCEWithLogitsLoss()

  def forward(
    self,
    logits: torch.Tensor,
    concept_scores: torch.Tensor | None,
    cosine_scores: torch.Tensor,
    targets: torch.Tensor,
    concept_targets: torch.Tensor,
    prototypes: torch.Tensor,
  ):
    loss_dict = dict(xe=self.xe(logits, targets))
    if concept_scores is None:
      if self.clst_coef != 0:
        loss_dict["clst"] = self.clst_coef * self.clst_criterion(cosine_scores, targets)
      if self.sep_coef != 0:
        loss_dict["sep"] = self.sep_coef * self.sep_criterion(cosine_scores, targets)
      if self.ortho_coef != 0:
        loss_dict["ortho"] = self.ortho_coef * self.ortho_criterion(prototypes)
    if self.bce_coef != 0 and concept_scores is not None:
      loss_dict["bce"] = self.bce_coef * self.bce(concept_scores, concept_targets.float())

    return sum(loss_dict.values()), loss_dict

  def clst_criterion(self, cosine_scores: torch.Tensor, targets: torch.Tensor):
    cosine_scores = cosine_scores.reshape(cosine_scores.size(0), self.num_classes, -1).max(dim=-1).values
    max_dist = 64
    positives = F.one_hot(targets, num_classes=self.num_classes).float()
    inverted_cosine_scores = (max_dist - cosine_scores) * positives
    min_cosine_scores = max_dist - inverted_cosine_scores.max(dim=-1).values
    return min_cosine_scores.mean()

  def sep_criterion(self, cosine_scores: torch.Tensor, targets: torch.Tensor):
    cosine_scores = cosine_scores.reshape(cosine_scores.size(0), self.num_classes, -1).max(dim=-1).values

    max_dist = 64
    negatives = 1 - F.one_hot(targets, num_classes=self.num_classes).float()
    inverted_cosine_scores = (max_dist - cosine_scores) * negatives
    min_cosine_scores = max_dist - inverted_cosine_scores.max(dim=-1).values
    return min_cosine_scores.mean()

  def ortho_criterion(self, prototypes: torch.Tensor):
    cur_basis_matrix = torch.squeeze(prototypes)
    subspace_basis_matrix = cur_basis_matrix.reshape(self.num_classes, 10, -1)
    subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix, 1, 2)
    ortho_operator = torch.matmul(subspace_basis_matrix, subspace_basis_matrix_T)
    I_operator = torch.eye(subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)).to(device=prototypes.device)
    difference_value = ortho_operator - I_operator
    ortho_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1, 2]) - 0))

    return ortho_cost
