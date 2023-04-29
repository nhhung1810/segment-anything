from glob import glob
import json
import os
import numpy as np
from torch import nn
import torch
from scripts.datasets.flare22_loader import FLARE22
from scripts.train.sam_train import SamTrain
from torch.utils.data import DataLoader

from segment_anything.build_sam import sam_model_registry
from scripts.train.loss import IoULoss, DiceLoss
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
def evaluate(sam_train: SamTrain, dataset: FLARE22, batch_size: int):
    # Eval need to activation
    iou_fn = IoULoss(activation=None, reduction="none")
    dice_fn = DiceLoss(activation=None, reduction="none")
    loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False)

    ious = []
    dices = []
    iou_mses = []
    # For one-mask prediction, we select the max-pred-iou
    select_indices = []

    input_size, original_size = dataset.get_size()
    for idx, batch in enumerate(loader):
        mask_pred, iou_pred, _ = sam_train.predict_torch(
            image_emb=batch["img_emb"],
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Eval need no logits
            return_logits=False,
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

        if idx == 2:
            break

    # Each of these will eventually have shape
    # of [dataset_length, 3]
    return make_metrics(ious, dices, iou_mses, select_indices)


if __name__ == "__main__":
    from pprint import PrettyPrinter

    pp = PrettyPrinter(indent=2)
    dataset = FLARE22(
        pre_trained_sam=None,
        metadata_path="dataset/FLARE22-version1/val_metadata.json",
        cache_name="simple-dataset/validation",
        is_debug=False,
        device='cuda:0'
    )
    
    # path = "./sam_vit_b_01ec64.pth"
    runs_dir = ...
    save = {}
    paths = glob(f"{runs_dir}/model-*.pt")
    model_entries = [
        {"name": "pretrain", "custom": None},
        *[{"name": os.path.basename(path), "custom": path} for path in paths]
    ]
    for entry in model_entries:
        model: Sam = sam_model_registry["vit_b"](
            checkpoint="./sam_vit_b_01ec64.pth", custom=entry['custom']
            )
        model.to('cuda:0')
        sam_train = SamTrain(sam_model=model)
        dataset.device = model.device
        dataset.preload()
        metrics = evaluate(sam_train=sam_train, dataset=dataset, batch_size=32)
        pp.print(metrics)
        save[entry['name']] = metrics

    with open("all_metrics.json") as out:
        json.dump(save, out)
    pass
