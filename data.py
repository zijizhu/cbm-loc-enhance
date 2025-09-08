import os
import pickle as pkl
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, Subset, random_split, default_collate
from torchvision import transforms as T
from torchvision.datasets import CelebA, ImageFolder

from data_celeba import generate_data, celeba_collate_fn


class SUNDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        split: str = "train",
        return_attributes: bool = False,
        return_img_path: bool = False,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        split_mat = loadmat((Path("data") / "SUN" / "splits.mat").as_posix(), squeeze_me=True)

        self.split_indices = {"train": split_mat["train_loc"] - 1, "val": split_mat["val_loc"] - 1, "test": split_mat["test_seen_loc"] - 1}
        self.split = split

        self.images = loadmat((Path(image_root) / "SUNAttributeDB" / "images.mat").as_posix(), squeeze_me=True)["images"].tolist()
        with open("data/SUN/sun_classes.txt") as fp:
            classes = fp.read().splitlines()
        self.classes = {c: i for i, c in enumerate(classes)}

        self.transforms = transforms
        self.image_root = image_root
        self.return_attributes = return_attributes
        self.return_img_path = return_img_path

    def __len__(self):
        return len(self.split_indices[self.split])

    def __getitem__(self, index: int):
        split_idx = self.split_indices[self.split][index]
        path = self.images[split_idx]

        class_name = path.split("/", 1)[-1].rsplit("/", 1)[0]
        class_name = " ".join(re.split("[_/]", class_name))
        img = Image.open(Path(self.image_root) / "images" / path).convert("RGB")
        class_idx = self.classes[class_name]

        return_data = [self.transforms(img), class_idx]
        if self.return_attributes:
            return_data.append(None)
        if self.return_img_path:
            return_data.append(img)

        return return_data


attribute_indices = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]

parts = ["head", "torso", "underparts"]


class CUBConceptDataset(ImageFolder):
    def __init__(
        self,
        image_root: str | Path,
        return_attributes: bool = False,
        return_img_path: bool = False,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root=image_root, transform=transforms, target_transform=target_transform)

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

        self.return_attributes = return_attributes
        self.return_img_path = return_img_path

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = Image.open(im_path).convert("RGB")
        im_pt = self.transform(im)
        attr = self.attributes[label]
        return_data = [im_pt, label]
        if self.return_attributes:
            return_data.append(attr)
        if self.return_img_path:
            return_data.append(im_path)
        return tuple(return_data)


def collate_fn_with_raw_images(batch):
    images, labels, attributes, raw_images = zip(*batch)

    image_batch = torch.stack(images, dim=0)
    label_batch = torch.tensor(labels)
    attribute_batch = torch.stack(attributes, dim=0)

    return image_batch, label_batch, attribute_batch, list(raw_images)


def generate_mask(
    prototype_to_part_logits: list[torch.Tensor], contrast_prototype_to_part_logits: list[torch.Tensor] | None, total_prototypes: int = 2000
):
    """
    shape of prototype_to_part_logits[i]: (num_prototypes_per_class, num_samples_class_i, num_parts)
    """
    prototype_to_part_indices = []
    if contrast_prototype_to_part_logits:
        for logits, contrast_logits in zip(prototype_to_part_logits, contrast_prototype_to_part_logits):
            proto_cpt_logits = (logits - contrast_logits).softmax(dim=-1).sum(dim=1)
            prototype_to_part_indices.append(proto_cpt_logits.argmax(dim=-1))
    else:
        for prototype_concept_logits in prototype_to_part_logits:
            proto_cpt_logits = prototype_concept_logits.softmax(dim=-1).sum(dim=1)
            prototype_to_part_indices.append(proto_cpt_logits.argmax(dim=-1))
    prototype_to_part_indices = torch.cat(prototype_to_part_indices, dim=0)

    with open("data/CUB/attribute_part_mapping.txt", "r") as fp:
        concept_to_part_mapping = fp.read().splitlines()

    concept_to_part_indices = [parts.index(p) if p in parts else -1 for p in concept_to_part_mapping]

    prototype_concept_mask = torch.full(
        (
            total_prototypes,
            len(concept_to_part_indices),
        ),
        False,
        dtype=torch.bool,
    )
    for c_idx, concept_part_idx in enumerate(concept_to_part_indices):
        for p_idx, prototype_part_idx in enumerate(prototype_to_part_indices):
            prototype_concept_mask[p_idx, c_idx] = concept_part_idx == prototype_part_idx

    return prototype_concept_mask.float()


#####################
# CelebA Dataset
#####################

CONCEPT_SEMANTICS = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class ModifiedCelebA(CelebA):
    def __init__(self, root: str, split="train", return_img_path=False, target_type: str | list[str] = "attr", transform=None, target_transform=None):
        super().__init__(root, split, target_type, transform, target_transform, False)
        self.return_img_path = return_img_path

    def __getitem__(
        self,
        index,
    ):
        image, (class_label, concept_labels) = super().__getitem__(index)
        if self.return_img_path:
            img_path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
            return image, class_label, concept_labels, img_path
        return image, class_label, concept_labels


def load_celeba_data(root_dir: str, transforms: Callable | None = None):
    # Process CelebA dataset per CEM
    # i.e. sort labels and select top 1000

    # config
    num_classes = 1000

    celeba_data = CelebA(
        root=os.path.expanduser(root_dir),
        split="all",
        download=False,
        target_type=["identity"],
    )

    classes, counts = np.unique(celeba_data.identity.numpy(), return_counts=True)
    sorted_indices = np.lexsort((classes, -counts))
    selected_classes = torch.from_numpy(np.take(classes, sorted_indices[:num_classes]))

    label_remap = {}
    for i, label in enumerate(selected_classes.tolist()):
        label_remap[label] = i

    def target_transform(labels: tuple[torch.Tensor, ...]):
        label, concepts = labels
        return (torch.tensor(label_remap.get(label.item(), num_classes), dtype=torch.long), concepts.float())

    celeba_data = ModifiedCelebA(
        root=os.path.expanduser(root_dir),
        split="all",
        transform=transforms,
        target_transform=target_transform,
        target_type=["identity", "attr"],
    )

    selected_sample_indices = torch.nonzero(torch.isin(celeba_data.identity.squeeze(), selected_classes)).squeeze()

    celeba_selected = Subset(
        celeba_data,
        selected_sample_indices,
    )

    total_samples = len(celeba_selected)
    train_samples = int(0.7 * total_samples)
    test_samples = total_samples - train_samples

    train_dataset, test_dataset = random_split(celeba_selected, [train_samples, test_samples])
    inference_dataset = deepcopy(train_dataset)
    inference_dataset.dataset.dataset.return_img_path = True

    return train_dataset, test_dataset, inference_dataset


def load_data(dataset_name: str, data_dir: str, batch_size: int, seed=42):
    assert dataset_name in ["CUB", "SUN", "CelebA"]
    transforms = T.Compose([
        T.Resize((
            224,
            224,
        )),
        T.ToTensor(),
        T.Normalize(
            (
                0.485,
                0.456,
                0.406,
            ),
            (
                0.229,
                0.224,
                0.225,
            ),
        ),
    ])

    if dataset_name == "SUN":
        train_dataset = SUNDataset(data_dir, split="train", transforms=transforms)
        inference_dataset = SUNDataset(data_dir, split="train", transforms=transforms)
        val_dataset = SUNDataset(data_dir, split="val", transforms=transforms)
        test_dataset = SUNDataset(data_dir, split="test", transforms=transforms)
        num_classes, num_concepts = 717, 102
        collate_fn = default_collate
    elif dataset_name == "CelebA":
        train_dataset, test_dataset, val_dataset, imbalance = generate_data(data_dir, resol=224, transform=transforms, seed=seed)
        inference_dataset = train_dataset
        num_classes, num_concepts = 256, 6
        collate_fn = celeba_collate_fn
    else:
        train_dataset = CUBConceptDataset(
            Path(data_dir) / "cub200_cropped" / "train_cropped_augmented", return_attributes=True, transforms=transforms
        )
        inference_dataset = CUBConceptDataset(
            Path(data_dir) / "cub200_cropped" / "train_cropped", transforms=transforms, return_attributes=True, return_img_path=True
        )
        test_dataset = CUBConceptDataset(Path(data_dir) / "cub200_cropped" / "test_cropped", return_attributes=True, transforms=transforms)
        num_classes, num_concepts = 200, 112
        collate_fn = default_collate
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    inference_loader = DataLoader(inference_dataset, collate_fn=collate_fn_with_raw_images, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader, inference_loader, num_classes, num_concepts
