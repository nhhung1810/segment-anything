from collections import defaultdict
from ftplib import all_errors
import os
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
from scripts.constants import IMAGE_TYPE
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.sam_train import SamTrain
from scripts.datasets.constant import VAL_METADATA, DEFAULT_DEVICE
from scripts.datasets.flare22_one_point import FileLoader
from scripts.utils import load_file_npz, load_img, make_directory, omit
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam
from scripts.losses.loss import DiceLoss


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
    return coors, labels


def frame_inference(
    sam_train: SamTrain,
    class_num: int,
    dice_fn: DiceLoss,
    img: np.ndarray,
    mask: np.ndarray,
    device: str = DEFAULT_DEVICE,
):
    mask = torch.as_tensor(mask == class_num)

    if mask.any():
        coors, labels = make_point_from_mask(mask=mask == class_num)
    else:
        coors, labels = None, None

    with torch.no_grad():
        img_emb, original_size, input_size = sam_train.prepare_img(image=img)

        coords_torch, labels_torch, _, _ = sam_train.prepare_prompt(
            original_size=original_size,
            point_coords=coors,
            point_labels=labels,
        )

        mask_pred, iou_pred, _ = sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Eval need no logits
            return_logits=False,
            point_coords=coords_torch,
            point_labels=labels_torch,
            mask_input=None,
        )

        mask = (
            mask.unsqueeze(0).repeat_interleave(3, dim=0).unsqueeze(0).type(torch.int64)
        )
        mask = mask.to(device=device)
        dice = dice_fn.run_on_batch(input=mask_pred, target=mask)
        chosen_idx = dice.argmin()
        mask_pred = mask_pred[0, chosen_idx]

        return mask_pred.detach().cpu().numpy()


def inference(
    images: List[str],
    gts: List[str],
    sam_train: SamTrain,
    inference_save_dir: str,
):
    class_num = 1
    dice_fn = DiceLoss(activation=None, reduction="none")
    preprocessor = FLARE22_Preprocess()
    for image_file, gt_file in zip(images, gts):
        patient_name = os.path.basename(gt_file).replace(".nii.gz", "")
        volumes, masks = preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )
        predict_volume = []
        for idx in range(volumes.shape[0]):
            img = volumes[idx, ...]
            mask = masks[idx, ...]
            pred = frame_inference(
                sam_train=sam_train,
                class_num=class_num,
                dice_fn=dice_fn,
                img=img,
                mask=mask,
            )
            predict_volume.append(pred)
            pass
        # Pack all prediction into volume
        predict_volume = np.stack(predict_volume, axis=0)

        mask_out_path = f"{inference_save_dir}/{patient_name}.npy"
        make_directory(mask_out_path, is_file=True)
        np.save(mask_out_path, predict_volume)

        pass


if __name__ == "__main__":
    file_loader = FileLoader(metadata_path=VAL_METADATA)
    patient_data = defaultdict(list)
    device = DEFAULT_DEVICE
    TEST_IMAGE_PATH = "dataset/FLARE22-version1/FLARE22_LabeledCase50/images/FLARE22_Tr_0008_0000.nii.gz"
    TEST_MASK_PATH = (
        "dataset/FLARE22-version1/FLARE22_LabeledCase50/labels/FLARE22_Tr_0008.nii.gz"
    )
    run_path = "sam-one-point-230501-012024"
    model_path = f"runs/{run_path}/model-10.pt"
    model = load_model(model_path)
    sam_train = SamTrain(sam_model=model)
    inference_save_dir = f"runs/{run_path}/inference/"

    inference(
        images=[TEST_IMAGE_PATH],
        gts=[TEST_MASK_PATH],
        inference_save_dir=inference_save_dir,
        sam_train=sam_train,
    )

    pass
