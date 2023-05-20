import os
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
from scripts.constants import FLARE22_LABEL_ENUM
from scripts.datasets.constant import DATASET_ROOT, IMAGE_TYPE, TRAIN_NON_PROCESSED
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


@torch.no_grad()
def frame_inference(
    sam_train: SamTrain,
    class_num: int,
    dice_fn: DiceLoss,
    img: np.ndarray,
    mask: np.ndarray,
    previous_mask: np.ndarray = None,
    device: str = DEFAULT_DEVICE,
    cache_img_emb_path: str = None,
    dice_with: str = "prev",
):
    mask = torch.as_tensor(mask == class_num)

    img_emb, original_size, input_size = run_prepare_img_with_cache(
        sam_train, img, cache_img_emb_path
    )

    if previous_mask is None:
        coors, labels = make_point_from_mask(mask)
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
        _, _, _, mask_input_torch = sam_train.prepare_prompt(
            original_size=original_size, mask_input=previous_mask[None, ...]
        )

        mask_pred, _, _ = sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Eval need no logits
            return_logits=True,
            mask_input=mask_input_torch,
        )

    mask_logit = mask_pred.clone()
    mask_pred = mask_pred > sam_train.model.mask_threshold
    stability_score = calculate_stability_score(
        masks=mask_logit,
        mask_threshold=sam_train.model.mask_threshold,
        threshold_offset=1.0,
    )

    # Dice loss -> take the smallest (the smaller the better)
    if dice_with == "gt":
        mask = (
            mask.unsqueeze(0).repeat_interleave(3, dim=0).unsqueeze(0).type(torch.int64)
        )
        mask = mask.to(device=device)
        dice = dice_fn.run_on_batch(input=mask_pred, target=mask)
        chosen_idx = dice.argmin()
    elif dice_with == "prev":
        previous_mask = torch.as_tensor(previous_mask)[
            None, None, ...
        ].repeat_interleave(3, dim=1)
        dice = dice_fn.run_on_batch(input=mask_pred, target=previous_mask)
        chosen_idx = dice.argmin()
        pass
    mask_pred = mask_pred[0, chosen_idx]
    stability_score = stability_score[0, chosen_idx].float()

    return mask_pred, dice.min().detach().item(), stability_score


def run_prepare_img_with_cache(sam_train, img, cache_img_emb_path):
    if cache_img_emb_path is None:
        img_emb, original_size, input_size = sam_train.prepare_img(image=img)
    else:
        _d = pick(
            torch.load(cache_img_emb_path, map_location="cpu"),
            ["img_emb", "original_size", "input_size"],
        )
        img_emb, original_size, input_size = (
            _d["img_emb"],
            _d["original_size"],
            _d["input_size"],
        )
        original_size = (original_size[0], original_size[1])
        input_size = (input_size[0], input_size[1])
    return img_emb, original_size, input_size


def visualize(
    inference_save_dir: str,
    class_num: int,
    gt_file: str,
    previous_mask: Tensor,
    idx: int,
    img: np.ndarray,
    mask: np.ndarray,
    pred: Tensor,
    dice_min: float,
    stability_score: float,
):
    render = Renderer(None)

    save_img_path = f"{inference_save_dir}/{os.path.basename(gt_file).replace('.nii.gz', '')}/{class_num}/{idx:0>4}.png"
    make_directory(save_img_path, is_file=True)

    render.add(img=img, mask=(mask == class_num).astype(np.uint8), title="GT").add(
        img=img,
        mask=pred.detach().cpu().numpy().astype(np.uint8),
        title=f"Best Pred: {dice_min:.2f} | {stability_score:.2f}",
    )
    if previous_mask is not None:
        _previous_mask = previous_mask
        if isinstance(previous_mask, Tensor):
            _previous_mask = previous_mask.detach().cpu().numpy()
        render.add(img=img, mask=_previous_mask.astype(np.uint8), title="Prev.")
        pass

    render.show_all(save_path=save_img_path).reset()


def inference(
    images: List[str],
    gts: List[str],
    sam_train: SamTrain,
    inference_save_dir: str,
    class_num: int,
):
    assert len(images) == len(gts)
    dice_fn = DiceLoss(activation=None, reduction="none")
    preprocessor = FLARE22_Preprocess()
    for image_file, gt_file in tqdm(
        zip(images, gts), total=len(images), desc="Inference for patient..."
    ):
        if "0008" not in gt_file:
            continue
        patient_name = os.path.basename(image_file).replace(".nii.gz", "")
        volumes, masks = preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )
        start_idx, end_idx = find_organ_range(masks=masks, class_num=class_num)
        predict_volume = []
        previous_mask = None
        for idx in tqdm(
            range(volumes.shape[0]),
            desc="Inference frame by frame...",
            leave=False,
            total=volumes.shape[0],
        ):
            if idx not in range(start_idx, end_idx + 1):
                # Heuristic: only predict inside the region
                save_pred = np.zeros(masks[idx, ...].shape, dtype=np.uint8)
            else:
                # Grey-scale 3 channels
                img = volumes[idx][..., None].repeat(3, -1)
                mask = masks[idx, ...]

                if previous_mask == None:
                    previous_mask = (mask == class_num).astype(np.uint8)

                pred, dice_min, stability_score = frame_inference(
                    sam_train=sam_train,
                    class_num=class_num,
                    dice_fn=dice_fn,
                    img=img,
                    mask=mask,
                    previous_mask=previous_mask,
                    cache_img_emb_path=f"{CACHE_DIR}/{patient_name}/{idx}.pt",
                )

                visualize(
                    inference_save_dir,
                    class_num,
                    gt_file,
                    previous_mask,
                    idx,
                    img,
                    mask,
                    pred,
                    dice_min,
                    stability_score,
                )

                previous_mask = pred
                save_pred = pred.detach().cpu().numpy().astype(np.uint8)
                pass
            predict_volume.append(save_pred)
            pass
        # Pack all prediction into volume
        predict_volume = np.stack(predict_volume, axis=0)
        # new shape=[H, W, T]
        predict_volume = predict_volume.transpose(1, 2, 0).astype(np.uint8)
        # To correct the class number
        predict_volume = predict_volume * class_num

        # Convert file into nii.gz format for inference
        post_process(
            pred=predict_volume,
            gt_file=gt_file,
            out_dir=inference_save_dir,
        )
        break

        pass


parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="Volume image directory",
    default=f"{TRAIN_NON_PROCESSED}/images",
)
parser.add_argument(
    "-l",
    "--label_dir",
    type=str,
    help="Volume label directory",
    default=f"{TRAIN_NON_PROCESSED}/labels",
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
    "-c", "--class_num", type=int, help="Class number to inference", default=2
)

if __name__ == "__main__":
    args = parser.parse_args()
    device = DEFAULT_DEVICE

    run_path = "mask-prop-230509-005503"
    model_path = args.checkpoint or f"runs/{run_path}/model-100.pt"
    model = load_model(model_path)
    sam_train = SamTrain(sam_model=model)

    inference_save_dir = args.output_dir
    input_dir = args.input_dir
    label_dir = args.label_dir
    class_num = args.class_num
    print(
        "🚀 ~ file: inference.py:173 ~ model_path:",
        model_path,
        class_num,
        inference_save_dir,
    )
    make_directory(inference_save_dir)
    assert (
        class_num < 14 and class_num > 0 and isinstance(class_num, int)
    ), f"Incorrect class num"

    images_path: List[str] = sorted(os.listdir(input_dir))
    labels_path = [
        os.path.join(label_dir, p.replace("_0000.nii.gz", ".nii.gz"))
        for p in images_path
    ]
    images_path = [os.path.join(input_dir, p) for p in images_path]

    for i, p in zip(images_path, labels_path):
        assert os.path.exists(i)
        assert os.path.exists(p)

    inference(
        images=images_path,
        gts=labels_path,
        inference_save_dir=inference_save_dir,
        sam_train=sam_train,
        class_num=class_num,
    )

    pass
