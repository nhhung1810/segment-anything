import os
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
from scripts.datasets.constant import IMAGE_TYPE, TRAIN_NON_PROCESSED
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.sam_train import SamTrain
from scripts.datasets.constant import DEFAULT_DEVICE

from scripts.tools.evaluation.loading import post_process
from scripts.utils import make_directory
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam
from scripts.losses.loss import DiceLoss
from argparse import ArgumentParser


def load_model(model_path, device=DEFAULT_DEVICE) -> Sam:
    model: Sam = sam_model_registry["vit_b"](
        checkpoint="./sam_vit_b_01ec64.pth", custom=model_path
    )
    model.to(device)
    return model


# def make_point_from_mask(mask: Tensor) -> Tuple[Tensor, Tensor]:
#     coors = torch.argwhere(mask)
#     coors = coors[torch.randperm(coors.shape[0])][:1]
#     labels = torch.ones(coors.shape[0], 1, 1)
#     if labels.shape[0] == 0:
#         return None, None
#     return coors, labels


def find_organ_range(masks: np.ndarray, class_num: int) -> Tuple[int, int]:
    start_idx, end_idx = None, None
    r = range(masks.shape[0])
    # [0, len, 1)
    for forward_idx, reverse_idx in zip(r, r[::-1]):
        if start_idx == None:
            start_idx = (masks[forward_idx, ...] == class_num).any()

        if end_idx == None:
            end_idx = (masks[reverse_idx, ...] == class_num).any()

        if start_idx != None and end_idx != None:
            return start_idx, end_idx

    return start_idx, end_idx


def frame_inference(
    sam_train: SamTrain,
    class_num: int,
    dice_fn: DiceLoss,
    img: np.ndarray,
    mask: np.ndarray,
    previous_mask: Tensor = None,
    device: str = DEFAULT_DEVICE,
):
    mask = torch.as_tensor(mask == class_num)

    with torch.no_grad():
        img_emb, original_size, input_size = sam_train.prepare_img(image=img)

        _, _, _, mask_input_torch = sam_train.prepare_prompt(
            original_size=original_size, mask_input=previous_mask
        )

        mask_pred, iou_pred, _ = sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Eval need no logits
            return_logits=False,
            mask_input=mask_input_torch,
        )

        mask = (
            mask.unsqueeze(0).repeat_interleave(3, dim=0).unsqueeze(0).type(torch.int64)
        )
        mask = mask.to(device=device)
        dice = dice_fn.run_on_batch(input=mask_pred, target=mask)
        chosen_idx = dice.argmin()
        mask_pred = mask_pred[0, chosen_idx]

        return mask_pred


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
        # patient_name = os.path.basename(gt_file).replace(".nii.gz", "")
        volumes, masks = preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )
        start_idx, end_idx = find_organ_range(volume=volumes, class_num=class_num)
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
                    previous_mask = mask

                pred = frame_inference(
                    sam_train=sam_train,
                    class_num=class_num,
                    dice_fn=dice_fn,
                    img=img,
                    mask=mask,
                    previous_mask=previous_mask,
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
        # mask_out_path = f"{inference_save_dir}/{patient_name}.npy"
        # make_directory(mask_out_path, is_file=True)
        # np.save(mask_out_path, predict_volume)

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

    run_path = ...
    model_path = args.checkpoint or f"runs/{run_path}/model-10.pt"
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
