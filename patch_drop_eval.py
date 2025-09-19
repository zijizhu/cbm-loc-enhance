import argparse
import logging
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torcheval.metrics import BinaryAccuracy
from torchvision import tv_tensors
from torchvision.transforms import v2
from tqdm import tqdm

from data import attribute_indices
from eval.local_parts import attributes_indexes
from nets import PPConceptNet


class CUBConceptDropDataset(Dataset):
    def __init__(self, data_root: str | Path, drop_attribute_index: int | None, crop_size: int = 50, check_integrity = False):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.drop_attribute_index = drop_attribute_index
        self.check_integrity = check_integrity
        with open(Path("data") / "CUB" / "class_attr_data_10" / "train.pkl", "rb") as fp:
            train_attribute_anns = pkl.load(fp)

        label2attr = dict()
        for ann in train_attribute_anns:
            label, attribute_vector = ann["class_label"], ann["attribute_label"]
            if label not in label2attr:
                label2attr[label] = attribute_vector

        self.attributes = torch.tensor([label2attr[i] for i in range(len(label2attr))], dtype=torch.long)

        with open(Path("data") / "CUB" / "cub_attributes_cleaned.txt", "r") as fp:
            all_attribute_texts = fp.read().splitlines()
        self.attribute_texts = [all_attribute_texts[i] for i in attribute_indices]

        images_df = pd.read_csv(
            Path(self.data_root) / "CUB_200_2011" / "images.txt",
            header=None,
            delimiter=" ",
            names=["img_id", "path"],
            usecols=[0, 1],
            index_col=0
        )
        splits_df = pd.read_csv(
            Path(self.data_root) / "CUB_200_2011" / "train_test_split.txt",
            header=None,
            delimiter=" ",
            names=["img_id", "is_train"],
            usecols=[0, 1],
            index_col=0
        )
        keypoints_df = pd.read_csv(
            Path(self.data_root) / "CUB_200_2011" / "parts"/ "part_locs.txt",
            header=None,
            delimiter=" ",
            names=["img_id", "part_idx", "x", "y"],
            usecols=[0, 1, 2, 3],
            index_col=0
        )
        bbox_df = pd.read_csv(
            Path(self.data_root) / "CUB_200_2011" / "bounding_boxes.txt",
            header=None,
            delimiter=" ",
            names=["img_id", "x", "y", "w", "h"],
            usecols=[0, 1, 2, 3, 4],
            index_col=0
        )
        images_df.index = images_df.index - 1
        splits_df.index = splits_df.index - 1
        keypoints_df.index = keypoints_df.index - 1
        keypoints_df["part_idx"] = keypoints_df["part_idx"] - 1
        bbox_df.index = bbox_df.index - 1

        self.images_df = images_df
        self.splits_df = splits_df
        self.keypoints_df = keypoints_df
        self.bbox_df = bbox_df

        self.samples_df = images_df.loc[splits_df["is_train"] == 0]

        # Process part name to idx mapping
        self.part_name2part_idx = defaultdict(list)
        with open(self.data_root / "CUB_200_2011" / "parts" / "parts.txt", "r") as fp:
            parts = [line.split(" ", 1)[1] for line in fp.read().splitlines()]
        for i, part_name in enumerate(parts):
            self.part_name2part_idx[part_name.split(" ")[-1]].append(i)

        with open(self.data_root / "CUB_200_2011" / "attributes.txt", "r") as fp:
            attributes = fp.read().splitlines()
            attributes = [attributes[i] for i in attributes_indexes]

        with open(self.data_root / "CUB_200_2011" / "parts" / "parts.txt", "r") as fp:
            parts = [line.split(" ", 1)[1] for line in fp.read().splitlines()]

        # Create a mapping from attribute index to part index
        self.attr_id2part_indices = defaultdict(list)
        for attr_idx, attr in enumerate(attributes):
            attr = attr.replace("bill", "beak")
            for part_name, part_indices in self.part_name2part_idx.items():
                if part_name in attr:
                    print("attr", f"{attr_idx}".ljust(3), " "*5, attr.split(" ")[1].ljust(40), "->", " " * 10, part_name)
                    self.attr_id2part_indices[attr_idx] += part_indices

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485,0.456,0.406,),(0.229,0.224,0.225,),),
        ])

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, index: int):
        im_id, im_path = self.samples_df.index[index], self.samples_df.iloc[index]["path"]
        label = int(im_path.split(".")[0]) - 1
        attr = self.attributes[label]

        im = Image.open(self.data_root / "cub200_cropped" / "test_cropped" / im_path)
        im = v2.functional.resize(im, [224, 224])

        if self.drop_attribute_index is None:
            return self.transforms(im), label, attr

        if self.check_integrity:
            assert attr[self.drop_attribute_index] == 1, f"Attribute to drop is not present as ground truth for sample {index}"

        sample_keypoints = self.keypoints_df.loc[
            (self.keypoints_df.index == im_id) &
            (self.keypoints_df["x"] != 0) &
            (self.keypoints_df["y"] != 0)
        ]
        raw_im = Image.open(self.data_root / "CUB_200_2011" / "images" / im_path)
        raw_w, raw_h = raw_im.size
        object_x, object_y, object_w, object_h = tuple(self.bbox_df.iloc[im_id][["x", "y", "w", "h"]])

        # Locate the part keypoints of the attribute, then apply crop and resize transforms on them
        part_indices = set(sample_keypoints['part_idx']) & set(self.attr_id2part_indices[self.drop_attribute_index])
        if len(part_indices) == 0:
            print("something wrong")
        part_cxcy = sample_keypoints.loc[sample_keypoints["part_idx"].isin(list(part_indices))][["x", "y"]].to_numpy()
        part_cxcy_pt = tv_tensors.KeyPoints(part_cxcy, canvas_size=(raw_h, raw_w))
        part_cxcy_transformed = v2.functional.crop(part_cxcy_pt, top=object_y, left=object_x, height=object_h, width=object_w)
        part_cxcy_transformed = v2.functional.resize(part_cxcy_transformed, [224, 224])

        draw = ImageDraw.Draw(im)
        for kp in part_cxcy_transformed:
            draw.rectangle(xy=[(kp - self.crop_size // 2).tolist(), (kp + self.crop_size // 2).tolist()], fill="black")
        im_pt = self.transforms(im)

        return im_pt, label, attr


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--epochs", type=int, default=9, help="Number of training epochs")
    parser.add_argument("--joint-start-epoch", type=int, default=3)
    parser.add_argument("--concept-layer-start-epoch", type=int, default=6)

    parser.add_argument("--concept-layer-only", action="store_true")
    parser.add_argument("--ckpt-path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=43)

    args = parser.parse_args()

    seed_everything(args.seed)
    torch.Generator().manual_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "patch_drop_eval.log").as_posix()),
        ],
        force=True,
    )

    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes, num_concepts = 200, 112
    model = PPConceptNet(
        backbone_name=args.backbone,
        num_classes=num_classes,
        k=args.k,
        num_concepts=num_concepts,
        use_mask=not args.disable_mask,
        use_basis_projection=not args.disable_basis_projection,
    )

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device=device)
    model.eval()

    dataset = CUBConceptDropDataset(
        data_root=Path(args.data_dir),
        drop_attribute_index=None
    )
    concept_drop_dataset = CUBConceptDropDataset(
        data_root=Path(args.data_dir),
        drop_attribute_index=0
    )

    attributes = dataset.attributes.clone().detach()

    images_df = dataset.images_df.copy()
    splits_df = dataset.splits_df.copy()

    images_df["label"] = images_df["path"].str.split(".").str[0].astype(int) - 1
    samples_df = pd.concat([images_df, splits_df], axis=1)
    samples_df = samples_df.loc[samples_df["is_train"] == 0]

    kp_df = dataset.keypoints_df.copy()
    kp_df = kp_df[(kp_df["x"] != 0) & (kp_df["y"] != 0)]

    sample_part_visible = np.zeros((len(dataset.images_df), 15))
    for i in range(len(dataset.images_df)):
        visible_part_indices = kp_df[kp_df.index == i]["part_idx"].to_numpy()
        sample_part_visible[i, visible_part_indices] = 1
    sample_part_visible = sample_part_visible[dataset.splits_df["is_train"] == 0].astype(bool)

    TOTAL_NUM_PARTS = 15

    statistics = []

    with torch.inference_mode():
        for attr_i in tqdm(concept_drop_dataset.attr_id2part_indices):
            attr2part = np.zeros(TOTAL_NUM_PARTS).astype(bool)
            attr2part[dataset.attr_id2part_indices[0]] = True

            attr_i_visible = (sample_part_visible & attr2part).sum(axis=1) > 0

            selected_class_indices = torch.nonzero(attributes[:, attr_i]).flatten().cpu().numpy()
            if selected_class_indices.size == 0:
                continue
            sample_mask = samples_df["label"].isin(selected_class_indices).to_numpy()
            sample_mask = sample_mask & attr_i_visible
            selected_sample_indices, = np.nonzero(sample_mask)

            concept_drop_dataset.drop_attribute_index = attr_i
            attr_i_drop_subset = Subset(dataset=concept_drop_dataset, indices=selected_sample_indices)
            attr_i_drop_loader = DataLoader(dataset=attr_i_drop_subset, batch_size=128, num_workers=8)

            attr_i_subset = Subset(dataset=dataset, indices=selected_sample_indices)
            attr_i_loader = DataLoader(dataset=attr_i_subset, batch_size=128, num_workers=8)

            correct, patch_drop_correct = 0, 0
            total, patch_drop_total = 0, 0

            bin_acc = BinaryAccuracy(threshold=0.7).to(device=device)
            patch_drop_bin_acc = BinaryAccuracy(threshold=0.7).to(device=device)

            for images, labels, attrs in attr_i_loader:
                images, labels, attributes = images.to(device), labels.to(device), attrs.to(device)
                logits, concept_scores, _, _, _ = model(images, with_concepts=True)

                predicted = torch.argmax(logits, dim=-1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                bin_acc.update(torch.sigmoid(concept_scores)[:, attr_i], attrs[:, attr_i])

            for images, labels, attrs in attr_i_drop_loader:
                images, labels, attributes = images.to(device), labels.to(device), attrs.to(device)
                logits, concept_scores, _, _, _ = model(images, with_concepts=True)

                predicted = torch.argmax(logits, dim=-1)
                patch_drop_correct += (predicted == labels).sum().item()
                patch_drop_total += labels.size(0)

                patch_drop_bin_acc.update(torch.sigmoid(concept_scores)[:, attr_i], attrs[:, attr_i])

            attr_i_stats = {
                "attr_idx": attr_i,
                "attr_name": dataset.attribute_texts[attr_i],
                "concept_acc": bin_acc.compute().cpu().item(),
                "class_acc": correct / total,
                "patch_drop_concept_acc": patch_drop_bin_acc.compute().cpu().item(),
                "patch_drop_class_acc": patch_drop_correct / patch_drop_total,
            }

            logger.info(f"{'' * 50}")
            for key, val in attr_i_stats.items():
                logger.info(f"{key.ljust(20)} {val}")

            statistics.append(attr_i_stats)

    stats_df = pd.DataFrame(statistics)
    stats_df.to_csv(log_dir / "patch_drop_stats.csv")
    mean_concept_acc_delta = stats_df['concept_acc'].mean() - stats_df['patch_drop_concept_acc'].mean()
    mean_class_acc_delta = stats_df['class_acc'].mean() - stats_df['patch_drop_class_acc'].mean()
    logger.info(f"Average concept accuracy delta over all attributes: {mean_concept_acc_delta}")
    logger.info(f"Average class accuracy delta over all attributes: {mean_class_acc_delta}")

if __name__ == "__main__":
    main()
