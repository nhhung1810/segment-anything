from glob import glob
import json
import os
import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from scripts.datasets.constant import VAL_METADATA
from scripts.datasets.flare22_one_point import FLARE22_One_Point
from scripts.sam_train import SamTrain
from torch.utils.data import DataLoader

from segment_anything.build_sam import sam_model_registry
from scripts.losses.loss import IoULoss, DiceLoss
from segment_anything.modeling.sam import Sam


def make_metrics(ious: list, dices: list, iou_mses: list, selected_indices):
    # Each of these have the shape of [N, 3]
    ious: np.ndarray = np.array(ious)
    dices: np.ndarray = np.array(dices)
    iou_mses: np.ndarray = np.array(iou_mses)

    # Basic mean
    mean_iou = ious.reshape(-1).mean()
    mean_dice = dices.reshape(-1).mean()
    mean_mse = iou_mses.reshape(-1).mean()

    # Selected iou mean -> get the predict_iou (max)
    mean_selected_iou = np.array(
        [ious[idx, selected_idx] for idx, selected_idx in enumerate(selected_indices)]
    ).mean()

    # Min/Max
    min_iou = ious.reshape(-1).min()
    min_dice = dices.reshape(-1).min()
    min_mse = iou_mses.reshape(-1).min()

    max_iou = ious.reshape(-1).max()
    max_dice = dices.reshape(-1).max()
    max_mse = iou_mses.reshape(-1).max()

    # Mean of best iou and dice for each triplet mask
    mean_best_iou = ious.max(axis=1).mean()
    mean_best_dice = dices.max(axis=1).mean()

    return {
        "iou/min": min_iou,
        "iou/max": max_iou,
        "iou/mean": mean_iou,
        "iou/mean_of_best": mean_best_iou,
        "iou/mean_of_selected": mean_selected_iou,
        "dice/min": min_dice,
        "dice/max": max_dice,
        "dice/mean": mean_dice,
        "dice/mean_of_best": mean_best_dice,
        "pred_iou_mse/min": min_mse,
        "pred_iou_mse/max": max_mse,
        "pred_iou_mse/mean": mean_mse,
    }


@torch.no_grad()
def evaluate(sam_train: SamTrain, dataset: FLARE22_One_Point):
    # Eval need to activation
    iou_fn = IoULoss(activation=None, reduction="none")
    dice_fn = DiceLoss(activation=None, reduction="none")
    loader = DataLoader(dataset=dataset, batch_size=1, drop_last=False)

    ious = []
    dices = []
    iou_mses = []
    # For one-mask prediction, we select the max-pred-iou
    select_indices = []

    input_size, original_size = dataset.get_size()
    for _, batch in enumerate(loader):
        coords_torch, labels_torch, _, _ = sam_train.prepare_prompt(
            original_size=original_size,
            point_coords=batch["coors"],
            point_labels=batch["labels"],
        )

        mask_pred, iou_pred, _ = sam_train.predict_torch(
            image_emb=batch["img_emb"],
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Eval need no logits
            return_logits=False,
            point_coords=coords_torch,
            point_labels=labels_torch,
            mask_input=None,
        )

        mask = batch["mask"]
        mask = mask.unsqueeze(1).repeat_interleave(3, dim=1).type(torch.int64)

        # result.shape = [batch_size, 3]
        iou = iou_fn.run_on_batch(input=mask_pred, target=mask)
        dice = dice_fn.run_on_batch(input=mask_pred, target=mask)

        # Invert the loss -> metrics
        iou = 1 - iou
        dice = 1 - dice

        # iou_mse = [batch_size, 3]
        iou_mse = (iou - iou_pred).square().mean(dim=1)

        ious.extend(iou.cpu().tolist())
        dices.extend(dice.cpu().tolist())
        iou_mses.extend(iou_mse.cpu().tolist())
        select_indices.extend(iou_pred.cpu().argmax(dim=1).tolist())

        # if idx == 2:
        #     break

    # Each of these will eventually have shape
    # of [dataset_length, 3]
    return make_metrics(ious, dices, iou_mses, select_indices)


if __name__ == "__main__":
    from pprint import PrettyPrinter

    pp = PrettyPrinter(indent=2)
    dataset = FLARE22_One_Point(
        pre_trained_sam=None,
        metadata_path=VAL_METADATA,
        cache_name=FLARE22_One_Point.VAL_CACHE_NAME,
        is_debug=False,
        device="cuda:0",
        coors_limit=1,
    )
    dataset.preprocess()

    # path = "./sam_vit_b_01ec64.pth"
    run_name = "sam-fix-iou-230429-011855"
    runs_dir = f"runs/{run_name}"
    save = {}
    paths = glob(f"{runs_dir}/model-*.pt")
    model_entries = [
        {"name": "pretrain", "custom": None},
        *[{"name": os.path.basename(path), "custom": path} for path in paths],
    ]
    for entry in tqdm(
        model_entries, desc="Running eval on checkpoint", total=len(model_entries)
    ):
        model: Sam = sam_model_registry["vit_b"](
            checkpoint="./sam_vit_b_01ec64.pth", custom=entry["custom"]
        )
        model.to("cuda:0")
        sam_train = SamTrain(sam_model=model)
        dataset.preload()
        metrics = evaluate(sam_train=sam_train, dataset=dataset, batch_size=1)
        save[entry["name"]] = metrics

    with open(f"{run_name}-all_metrics.json", "w") as out:
        json.dump(save, out)
    pass
