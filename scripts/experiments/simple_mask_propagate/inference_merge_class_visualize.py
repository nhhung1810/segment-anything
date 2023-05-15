from glob import glob
import os
from typing import Dict, List, Optional, Tuple

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
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.experiments.simple_mask_propagate.inference_merge_class import (
    load_model,
    get_all_organ_range,
    pick_best_mask,
    merge_function,
)
from scripts.render.data_class import (
    ImageData,
    MaskData,
    MultiMasksData,
    OneImageRenderData,
)
from scripts.render.render_engine import RenderEngine
from scripts.sam_train import SamTrain
from scripts.datasets.constant import DEFAULT_DEVICE

from scripts.tools.evaluation.loading import post_process
from scripts.utils import make_directory
from scripts.utils import torch_try_load
from segment_anything.modeling.sam import Sam
from scripts.losses.loss import DiceLoss
from argparse import ArgumentParser
import matplotlib.pyplot as plt


LEGEND_DICT = {
    value.value: value.name.lower().replace("_", " ") for value in FLARE22_LABEL_ENUM
}

DICE_FN = DiceLoss(activation=None, reduction="none")


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
    return merged_mask, class_pred_list, cache_frame


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

    # Get the previous mask using GT data if don't have
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


def inference(
    images: List[str],
    gts: List[str],
    sam_train: SamTrain,
    inference_save_dir: str,
    selected_class: List[int],
    device: str,
    visual_dir: str,
):
    assert len(images) == len(gts)
    preprocessor = FLARE22_Preprocess()
    for image_file, gt_file in tqdm(
        zip(images, gts), total=len(images), desc="Inference for patient..."
    ):
        patient_name = os.path.basename(image_file).replace(".nii.gz", "")
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
            # Predict and merge by frame
            merged_prediction, class_pred_dict, cache_frame = frame_inference(
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

            # Prepare for next prediction
            previous_mask = merged_prediction
            # Save data
            save_pred = merged_prediction.detach().cpu().numpy().astype(np.uint8)
            predict_volume.append(save_pred)
            # Cache
            cache_volume[idx] = cache_frame
            # Render
            visualize(
                patient_name=patient_name,
                visual_dir=visual_dir,
                idx=idx,
                img=img,
                merged_prediction=merged_prediction,
                class_pred_dict=class_pred_dict,
                ground_truth_mask=mask,
            )
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


def visualize(
    patient_name,
    visual_dir,
    idx,
    img,
    ground_truth_mask: np.ndarray,
    merged_prediction=None,
    class_pred_dict: Dict[int, Tensor] = None,
):
    class_pred = []
    for _, v in class_pred_dict.items():
        class_pred.append(v.detach().cpu().numpy())
    class_pred.append(ground_truth_mask == 1)
    class_pred.append(ground_truth_mask == 9)
    # multi_mask = np.stack(class_pred).astype(np.uint8)
    r = RenderEngine()
    (
        r.add(
            OneImageRenderData(
                image=ImageData(img),
                multi_masks=MultiMasksData(
                    masks=np.stack(class_pred[:2]).astype(np.uint8), legend=None
                ),
            )
        )
        .add(
            OneImageRenderData(
                image=ImageData(img),
                multi_masks=MultiMasksData(
                    masks=np.stack(class_pred[2:]).astype(np.uint8),
                    legend=["liver", "gall"],
                ),
            )
        )
        .show(save_path=f"{visual_dir}/{patient_name}_{idx:0>4}.png")
        .reset()
    )
    return


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
    default=[1, 9],  # for liver and gallbladder
)

if __name__ == "__main__":
    args = parser.parse_args()
    device = f"cuda:{args.cuda}" if "cuda" in DEFAULT_DEVICE else DEFAULT_DEVICE

    run_path = "mask-prop-230509-005503"
    model_path = args.checkpoint or f"runs/{run_path}/model-100.pt"
    visual_dir = f"runs/visualize/{run_path}/{os.path.basename(model_path)}/"
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
    # images_path = [i for i in images_path if "0002" in i]
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
    images_path = [os.path.basename(p) for p in images_path]
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
        visual_dir=visual_dir,
    )

    pass
