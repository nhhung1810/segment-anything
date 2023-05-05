from collections import defaultdict
from ftplib import all_errors
from typing import Tuple
import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
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


if __name__ == "__main__":
    file_loader = FileLoader(metadata_path=VAL_METADATA)
    patient_data = defaultdict(list)
    device = DEFAULT_DEVICE

    TEST_MASK_PATH = (
        "dataset/FLARE22-version1/FLARE22_LabeledCase50/labels/FLARE22_Tr_0008.nii.gz"
    )
    run_path = "sam-one-point-230501-012024"
    model_path = f"runs/{run_path}/model-10.pt"
    model = load_model(model_path)
    sam_train = SamTrain(sam_model=model)
    inference_save_dir = f"runs/{run_path}/inference/"

    # Liver only
    class_num = 1
    dice_fn = DiceLoss(activation=None, reduction="none")

    for data in file_loader.data:
        patient_data[data["name"]].append(omit(data, ["name"]))
        pass

    for patient_name, patient_val in tqdm(patient_data.items(), desc="Inference..."):
        if "FLARE22_Tr_0008" not in patient_name:
            continue

        all_pred = []
        for ct_slice in tqdm(
            patient_val,
            total=len(patient_val),
            leave=False,
            desc="Per-frame running...",
        ):
            img = load_img(ct_slice["img_path"])
            mask = torch.as_tensor(load_file_npz(ct_slice["mask_path"]))
            mask = mask == class_num
            if mask.any():
                coors, labels = make_point_from_mask(mask=mask == class_num)
            else:
                coors, labels = None, None

            # img_emb
            with torch.no_grad():
                img_emb, original_size, input_size = sam_train.prepare_img(image=img)

                coords_torch, labels_torch, _, _ = sam_train.prepare_prompt(
                    original_size=original_size,
                    point_coords=coors,
                    point_labels=labels,
                )

                mask_pred, iou_pred, _ = sam_train.predict_torch(
                    image_emb=img_emb, # 1, 256, 64, 64
                    input_size=input_size,
                    original_size=original_size,
                    multimask_output=True,
                    # Eval need no logits
                    return_logits=False,
                    point_coords=coords_torch,
                    point_labels=labels_torch,
                    mask_input=None,
                )
                
                mask = mask.unsqueeze(0).repeat_interleave(3, dim=0).unsqueeze(0).type(torch.int64)
                mask = mask.to(device=device)
                dice = dice_fn.run_on_batch(input=mask_pred, target=mask)
                chosen_idx = dice.argmin()
                mask_pred = mask_pred[0, chosen_idx]

                all_pred.append(mask_pred.detach().cpu().numpy())

                pass
        # Stack all prediction into one
        all_pred = np.stack(all_pred, axis=0)

        # FIXME: Check if need to transpose
        # H, W, T
        all_pred = all_pred.transpose(1, 2, 0).astype(np.uint8)
        mask_out_path = f"{inference_save_dir}/{patient_name}.npy"
        make_directory(mask_out_path, is_file=True)
        np.save(mask_out_path, all_pred)
        pass

    pass
