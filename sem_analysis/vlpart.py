import numpy as np
from pathlib import Path
import torch
import detectron2.data.transforms as T
from torch.nn import functional as F
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from VLPart.vlpart.config import add_vlpart_config
from VLPart.vlpart.modeling.meta_arch.vlm_rcnn_inference import *  # noqa: F403
from detectron2.data import MetadataCatalog

from detectron2.structures import Instances, Boxes

BUILTIN_CLASSIFIER = {
    'pascal_part': 'datasets/metadata/pascal_part_clip_RN50_a+cname.npy',
    'partimagenet': 'datasets/metadata/partimagenet_clip_RN50_a+cname.npy',
    'paco': 'datasets/metadata/paco_clip_RN50_a+cname.npy',
    'lvis': 'datasets/metadata/lvis_v1_clip_RN50_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_RN50_a+cname.npy',
    'voc': 'datasets/metadata/voc_clip_RN50_a+cname.npy',
}

def get_activation_bboxes_scaled(activation_maps: torch.Tensor,
                                 threshold: float,
                                 image_height: int,
                                 image_width: int,
                                 eps: float = 1e-6):
    assert activation_maps.ndim == 4
    batch_size, num_channels, act_height, act_width = activation_maps.shape
    bboxes = []

    activation_maps = F.interpolate(activation_maps, size=(image_height, image_width), mode="bilinear")

    for b in range(batch_size):
        sample_bboxes = []
        for c in range(num_channels):
            act_map = activation_maps[b, c]

            # Min-max normalize
            min_val = act_map.min()
            max_val = act_map.max()
            norm_map = (act_map - min_val) / (max_val - min_val + eps)

            # Threshold
            mask = norm_map > threshold

            if mask.any():
                ys, xs = torch.nonzero(mask, as_tuple=True)
                x_min = int(xs.min().item())
                x_max = int((xs.max().item()))
                y_min = int(ys.min().item())
                y_max = int((ys.max().item()))
                sample_bboxes.append([x_min, y_min, x_max, y_max])
            else:
                bboxes.append([-1, -1, -1, -1])  # No region exceeds threshold

        bboxes.append(torch.tensor(sample_bboxes))

    return bboxes


def setup_cfg(config_file: str | Path, opts: list[str], confidence_threshold: float):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_vlpart_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST =confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def reset_cls_test(model, cls_path):
    if isinstance(cls_path, str):
        print('Resetting zs_weight', cls_path)
        if cls_path.endswith('npy'):
            zs_weight = np.load(cls_path)
            zs_weight = torch.tensor(zs_weight, dtype=torch.float32).permute(1, 0).contiguous()  # dim x C
        elif cls_path.endswith('pth'):
            zs_weight = torch.load(cls_path, map_location='cpu')
            zs_weight = zs_weight.clone().detach().permute(1, 0).contiguous()  # dim x C
        else:
            raise NotImplementedError
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))],
        dim=1) # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)

    if isinstance(model.roi_heads.box_predictor, torch.nn.ModuleList):
        for idx in range(len(model.roi_heads.box_predictor)):
            model.roi_heads.box_predictor[idx].cls_score.zs_weight_inference = zs_weight
    else:
        model.roi_heads.box_predictor.cls_score.zs_weight_inference = zs_weight


BIRD_PART_START_IDX = 7
BIRD_PART_END_IDX = 16

CELEB_PART_START_IDX = 70
CELEB_PART_END_IDX = 76

bird_idx2part = {
    0: "head",
    1: "head",
    2: "head",
    3: "leg",
    4: "leg",
    5: "wing",
    6: "neck",
    7: "tail",
    8: "torso"
}

celeba_idx2part = {
    0: "hair",
    1: "head",
    2: "ear",
    3: "eye",
    4: "nose",
    5: "neck",
    6: "mouth"
}

class BatchedPredictorWithProposal:
    """Adapted from detectron 2 DefaultPredictor"""
    def __init__(self, cfg, dataset: str = "CUB"):
        assert dataset in ["CUB", "CelebA"]
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # Changed from ResizeShortestEdge to Resize
        self.aug = T.Resize([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST])

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        if dataset == "CUB":
            self.start_idx = BIRD_PART_START_IDX
            self.end_idx = BIRD_PART_END_IDX
        else:
            self.start_idx = CELEB_PART_START_IDX
            self.end_idx = CELEB_PART_END_IDX

    def __call__(self, original_images: list, proposals: list[Instances] | None = None):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_images = [img[:, :, ::-1] for img in original_images]
            batched_inputs = []
            for original_img in original_images:
                height, width = original_img.shape[:2]
                image = self.aug.get_transform(original_img).apply_image(original_img)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                image.to(self.cfg.MODEL.DEVICE)

                inputs = {"image": image, "height": height, "width": width}
                batched_inputs.append(inputs)

            predictions = self.model(batched_inputs, proposals)
        return [pred[..., self.start_idx:self.end_idx] for pred in predictions]


@torch.no_grad()
def generate_prototype_semantics(model, dataloader, save_dir: Path | str = "./", threshold: float = 0.7, k: int = 10,
                                 dataset_name: str = "CUB", num_classes: int = 200, device: str = "cuda"):
    model.eval()

    cfg = setup_cfg("VLPart/configs/others/r50_pascalpart_inference.yaml",
                    opts=["MODEL.WEIGHTS", "checkpoints/r50_pascalpart.pth",
                          "VIS.BOX", False,
                          "MODEL.DEVICE", device,
                          "MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH",
                          "VLPart/datasets/metadata/pascal_part_clip_RN50_a+cname.npy",
                          "MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH",
                          "VLPart/datasets/metadata/pascal_part_clip_RN50_a+cname.npy"],
                    confidence_threshold=0.7)
    detector = BatchedPredictorWithProposal(cfg, dataset=dataset_name)
    classifier_weight_path = os.path.join("VLPart", BUILTIN_CLASSIFIER["pascal_part"])
    reset_cls_test(detector.model, classifier_weight_path)

    prototype_semantics_scores = {cls: [] for cls in range(num_classes)}  # type: dict[int, list]

    for batch in tqdm(dataloader, total=len(dataloader)):
        images, labels, attributes, raw_image_paths = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)
        logits, concept_scores, cosine_scores, cosine_activations, activations = model(images, with_concepts=True)

        h = w = activations.size(-1)
        batch_size = logits.size(0)
        gt_activations = activations.reshape(batch_size, num_classes, -1, h, w)[torch.arange(batch_size), labels]

        raw_images = [read_image(path, format="BGR") for path in raw_image_paths]

        batched_boxes = get_activation_bboxes_scaled(gt_activations,
                                                     threshold,
                                                     cfg.INPUT.MIN_SIZE_TEST,
                                                     cfg.INPUT.MIN_SIZE_TEST)

        proposals = [
            Instances(image_size=(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST,),
                      proposal_boxes=Boxes(boxes),
                      objectness_logits=torch.full((boxes.size(0),), 10)).to(device)
            for boxes in batched_boxes
        ]

        predictions = detector(raw_images, proposals)

        for pred, cls in zip(predictions, labels.tolist()):
            prototype_semantics_scores[cls].append(pred.cpu())

    prototype_part_mapping = []
    idx2part = bird_idx2part if dataset_name == "bird" else celeba_idx2part
    for cls_i, cls_i_logits in prototype_semantics_scores.items():
        logits = torch.stack(cls_i_logits, dim=1)
        mean_logits = logits.softmax(dim=-1).mean(dim=1)
        part_indices = mean_logits.argmax(dim=-1)

        prototype_part_mapping.extend([idx2part[idx] for idx in part_indices.tolist()])

    attr_part_map_fn = 'data/CUB/attribute_part_mapping.txt' if dataset_name == "CUB" else "./data/CelebA/attribute_part_mapping.txt"

    with open(attr_part_map_fn, 'r') as fp:
        concept_part_mapping = fp.read().splitlines()

    num_concepts = len(concept_part_mapping)

    prototype_concept_mask = torch.full((num_classes * k, num_concepts,), False, dtype=torch.bool)
    for p_idx, proto_part_name in enumerate(prototype_part_mapping):
        for c_idx, cpt_part_name in enumerate(concept_part_mapping):
            prototype_concept_mask[p_idx, c_idx] = proto_part_name == cpt_part_name

    torch.save(dict(prototype_semantics_scores=prototype_semantics_scores,
                    prototype_concept_mask=prototype_concept_mask),
               Path(save_dir) / "prototype_semantics.pth")

    return prototype_concept_mask.float()
