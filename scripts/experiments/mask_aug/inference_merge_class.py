from glob import glob
import os
from typing import Dict, List, Optional, Tuple
from git import CacheError
import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
from scripts.constants import FLARE22_LABEL_ENUM
from scripts.datasets.constant import (
    DATASET_ROOT,
    IMAGE_TYPE,
    TRAIN_NON_PROCESSED,
    TEST_NON_PROCESSED,
)
from scripts.datasets.flare22_simple_mask_propagate import FLARE22_SimpleMaskPropagate
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.render.render import Renderer
from scripts.sam_train import SamTrain
from scripts.datasets.constant import DEFAULT_DEVICE

from scripts.tools.evaluation.loading import post_process
from scripts.utils import make_directory, pick
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam
from scripts.losses.loss import DiceLoss
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from segment_anything.utils.amg import calculate_stability_score


CACHE_DIR = f"{DATASET_ROOT}/{FLARE22_SimpleMaskPropagate.VAL_CACHE_NAME}/"
LEGEND_DICT = {
    value.value: value.name.lower().replace("_", " ") for value in FLARE22_LABEL_ENUM
}

DICE_FN = DiceLoss(activation=None, reduction="none")


def load_model(model_path, device=DEFAULT_DEVICE) -> Sam:
    model: Sam = sam_model_registry["vit_b"](
        checkpoint="./sam_vit_b_01ec64.pth", custom=model_path
    )
    model.to(device)
    return model


def make_point_from_mask(mask: Tensor) -> Tuple[Tensor, Tensor]:
    coors = torch.argwhere(mask)
    coors = coors[torch.randperm(coors.shape[0])][:1]
    labels = torch.ones(coors.shape[0], 1, 1)
    if labels.shape[0] == 0:
        return None, None
    return coors, labels


def find_organ_range(masks: np.ndarray, class_num: int) -> Tuple[int, int]:
    start_idx, end_idx = None, None
    r = range(masks.shape[0])
    # [0, len, 1)
    for forward_idx, reverse_idx in zip(r, r[::-1]):
        if start_idx == None:
            start_idx = (
                forward_idx if (masks[forward_idx, ...] == class_num).any() else None
            )

        if end_idx == None:
            end_idx = (
                reverse_idx if (masks[reverse_idx, ...] == class_num).any() else None
            )

        if start_idx != None and end_idx != None:
            return start_idx, end_idx

    return start_idx, end_idx


def get_all_organ_range(masks):
    starts, ends = [], []
    for class_num in range(1, 14, 1):
        start_idx, end_idx = find_organ_range(masks=masks, class_num=class_num)
        starts.append(start_idx)
        ends.append(end_idx)

    # Padding zero for background pixel
    return np.array([0, *starts]), np.array([0, *ends])


def pick_best_mask(
    pred_multi_mask: Tensor,
    previous_mask: np.ndarray,
    gt_binary_mask: Tensor,
    device: str,
    strategy: str,
) -> Tuple[int, float]:
    if strategy == "gt":
        gt_binary_mask = (
            gt_binary_mask.unsqueeze(0)
            .repeat_interleave(3, dim=0)
            .unsqueeze(0)
            .type(torch.int64)
            .to(device)
        )
        dice = DICE_FN.run_on_batch(input=pred_multi_mask, target=gt_binary_mask)
        chosen_idx = dice.argmin()
        return chosen_idx.int(), dice.detach().cpu().item()

    if strategy == "prev":
        previous_mask = torch.as_tensor(previous_mask, device=device)[
            None, None, ...
        ].repeat_interleave(3, dim=1)
        dice = DICE_FN.run_on_batch(input=pred_multi_mask, target=previous_mask)
        chosen_idx = dice.argmin()
        return chosen_idx, dice.min().detach().cpu().item()


# Merge by min-area
def merge_function(
    class_pred_list: Dict[int, torch.Tensor],
    default_mask_size=(512, 512),
    device: str = DEFAULT_DEVICE,
):
    area = {}
    total_area = float(default_mask_size[0] * default_mask_size[1])
    selected_class = list(class_pred_list.keys())
    for class_num in selected_class:
        class_mask = class_pred_list.get(class_num)
        area[class_num] = class_mask.sum().detach().cpu().float() / total_area
        pass
    # Sorted by value, descending -> min value have higher priority
    # therefore, will be add to the mask later -> overwrite them
    result = torch.zeros(default_mask_size, device=device)
    priority = sorted(area.items(), key=lambda x: x[1], reverse=True)
    for class_num, _ in priority:
        mask = class_pred_list[class_num]
        result[mask > 0.0] = class_num
        pass
    return result


@torch.no_grad()
def frame_inference(
    sam_train: SamTrain,
    selected_class: List[int],
    frame_idx: int,
    img: np.ndarray,
    mask: np.ndarray,
    class_appearance_range: Tuple[np.ndarray, np.ndarray],
    cache_frame: Optional[Dict[str, Tensor]] = None,
    previous_merged_mask: np.ndarray = None,
    device: str = DEFAULT_DEVICE,
    dice_with: str = "prev",
):
    if cache_frame is None:
        img_emb, original_size, input_size = sam_train.prepare_img(image=img)
        cache_frame = dict(
            img_emb=img_emb, original_size=original_size, input_size=input_size
        )
        pass
    img_emb = cache_frame["img_emb"]
    original_size = cache_frame["original_size"]
    input_size = cache_frame["input_size"]

    starts, ends = class_appearance_range
    class_pred_list = {}
    for class_num in selected_class:
        if frame_idx not in range(int(starts[class_num]), int(ends[class_num])):
            class_pred_list[class_num] = torch.zeros(original_size, device=device)
            continue

        mask_pred = class_inference(
            sam_train,
            mask,
            previous_merged_mask,
            device,
            dice_with,
            img_emb,
            original_size,
            input_size,
            class_num,
        )
        class_pred_list[class_num] = mask_pred
        pass

    merged_mask = merge_function(class_pred_list, device=device)
    return merged_mask, cache_frame


def class_inference(
    sam_train: SamTrain,
    mask: np.ndarray,
    previous_merged_mask: np.ndarray,
    device: str,
    dice_with: str,
    img_emb: Tensor,
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    class_num: int,
):
    class_mask = torch.as_tensor(mask == class_num)
    if False:
        # if previous_merged_mask is None or (not class_mask.any()):
        coors, labels = make_point_from_mask(class_mask)
        coords_torch, labels_torch, _, _ = sam_train.prepare_prompt(
            original_size=original_size,
            point_coords=coors,
            point_labels=labels,
        )

        mask_pred, _, _ = sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Eval need no logits
            return_logits=True,
            point_coords=coords_torch,
            point_labels=labels_torch,
        )
        pass
    else:
        # There are no previous mask for this class -> use the ground truth
        if previous_merged_mask is None:
            previous_mask = mask == class_num
        elif not (previous_merged_mask == class_num).any():
            previous_mask = mask == class_num
        else:
            previous_mask = previous_merged_mask == class_num
        _, _, _, mask_input_torch = sam_train.prepare_prompt(
            original_size=original_size, mask_input=previous_mask[None, ...]
        )

        mask_pred, _, _ = sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Get the logit to compute the stability
            return_logits=False,
            mask_input=mask_input_torch,
        )

    # Dice loss -> take the smallest (the smaller the better)
    chosen_idx, _ = pick_best_mask(
        pred_multi_mask=mask_pred,
        previous_mask=previous_mask,
        gt_binary_mask=class_mask,
        device=device,
        strategy=dice_with,
    )
    # Pick the mask and add to dict
    mask_pred = mask_pred[0, chosen_idx]
    return mask_pred


def torch_try_load(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device)
    except Exception as msg:
        pass
    return {}


def inference(
    images: List[str],
    gts: List[str],
    sam_train: SamTrain,
    inference_save_dir: str,
    selected_class: List[int],
    device: str,
):
    assert len(images) == len(gts)
    preprocessor = FLARE22_Preprocess()
    for image_file, gt_file in tqdm(
        zip(images, gts), total=len(images), desc="Inference for patient..."
    ):
        # patient_name = os.path.basename(image_file).replace(".nii.gz", "")
        cache_volume_path = image_file.replace(".nii.gz", ".cache.pt")
        volumes, masks = preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )

        starts, ends = get_all_organ_range(masks)
        cache_volume = torch_try_load(cache_volume_path, device=device)
        predict_volume = []
        previous_mask = None
        for idx in tqdm(
            range(volumes.shape[0]),
            desc="Inference frame by frame...",
            leave=False,
            total=volumes.shape[0],
        ):
            # Grey-scale 3 channels
            img = volumes[idx][..., None].repeat(3, -1)
            mask = masks[idx, ...]
            cache_frame = cache_volume.get(idx, None)
            merged_prediction, cache_frame = frame_inference(
                sam_train=sam_train,
                selected_class=selected_class,
                class_appearance_range=(starts, ends),
                frame_idx=idx,
                img=img,
                mask=mask,
                previous_merged_mask=previous_mask,
                device=device,
                cache_frame=cache_frame,
            )

            previous_mask = merged_prediction
            save_pred = merged_prediction.detach().cpu().numpy().astype(np.uint8)
            predict_volume.append(save_pred)
            cache_volume[idx] = cache_frame
            pass
        # Pack all prediction into volume
        predict_volume = np.stack(predict_volume, axis=0)
        # new shape=[H, W, T]
        predict_volume = predict_volume.transpose(1, 2, 0).astype(np.uint8)

        # Convert file into nii.gz format for inference
        post_process(
            pred=predict_volume,
            gt_file=gt_file,
            out_dir=inference_save_dir,
        )
        torch.save(cache_volume, cache_volume_path)
        pass


parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="Volume image directory",
    default=f"{TEST_NON_PROCESSED}/images",
)
parser.add_argument(
    "--cuda",
    type=int,
    help="GPU index",
    default=1,
)
parser.add_argument(
    "-l",
    "--label_dir",
    type=str,
    help="Volume label directory",
    default=f"{TEST_NON_PROCESSED}/labels",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Output directory",
    default="runs/submission/pred",
)
parser.add_argument(
    "-ckpt", "--checkpoint", type=str, help="Model checkpoint to load", default=None
)
parser.add_argument(
    "--use_cache",
    type=str,
    help="Allow the model to use embedding cache for faster inference",
    default=True,
)

parser.add_argument(
    "--selected_class",
    nargs="+",
    type=int,
    help="List of class, no pre/pos-fix separated by space, i.e. 1 2 3",
    default=None,  # for liver and gallbladder
)

if __name__ == "__main__":
    args = parser.parse_args()
    device = f"cuda:{args.cuda}" if "cuda" in DEFAULT_DEVICE else DEFAULT_DEVICE

    run_path = "mask-prop-230509-005503"
    model_path = args.checkpoint or f"runs/{run_path}/model-100.pt"
    model = load_model(model_path, device)
    sam_train = SamTrain(sam_model=model)

    inference_save_dir = args.output_dir
    input_dir = args.input_dir
    label_dir = args.label_dir
    selected_class = args.selected_class or list(range(1, 14))
    use_cache = args.use_cache
    for c in selected_class:
        assert c in range(1, 14)
    print(
        f"""
        model-path     : {model_path}
        save-dir       :  {inference_save_dir}
        input-dir      : {input_dir}
        label-dir      : {label_dir}
        is-use-cache   : {use_cache}
        selected-class : {selected_class}
        """
    )
    make_directory(inference_save_dir)

    images_path: List[str] = sorted(glob(f"{input_dir}/*.nii.gz"))
    images_path = [os.path.basename(p) for p in images_path]
    # By some way, they don't have gallbladder, which i will omit for now
    images_path.remove("FLARETs_0006_0000.nii.gz")
    images_path.remove("FLARETs_0008_0000.nii.gz")
    images_path.remove("FLARETs_0021_0000.nii.gz")
    images_path.remove("FLARETs_0031_0000.nii.gz")
    images_path.remove("FLARETs_0033_0000.nii.gz")
    images_path.remove("FLARETs_0036_0000.nii.gz")
    images_path.remove("FLARETs_0038_0000.nii.gz")
    images_path.remove("FLARETs_0043_0000.nii.gz")
    images_path.remove("FLARETs_0044_0000.nii.gz")
    images_path.remove("FLARETs_0048_0000.nii.gz")
    labels_path = [
        os.path.join(label_dir, p.replace("_0000.nii.gz", ".nii.gz"))
        for p in images_path
    ]
    images_path = [os.path.join(input_dir, p) for p in images_path]

    for i, p in zip(images_path, labels_path):
        assert os.path.exists(i), f"{i}"
        assert os.path.exists(p), f"{p}"

    inference(
        images=images_path,
        gts=labels_path,
        inference_save_dir=inference_save_dir,
        sam_train=sam_train,
        selected_class=selected_class,
        device=device,
    )

    pass
