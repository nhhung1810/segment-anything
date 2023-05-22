import json
from pprint import PrettyPrinter

from tqdm import tqdm
from scripts.datasets.constant import VAL_METADATA
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry
import torch
from torch import Tensor
from scripts.datasets.flare22_simple_mask_propagate import FLARE22_SimpleMaskPropagate
from scripts.sam_train import SamTrain
from scripts.losses.loss import IoULoss, DiceLoss
from torch.utils.data import DataLoader
import numpy as np


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


def evaluate(sam_train: SamTrain, dataset: FLARE22_SimpleMaskPropagate, batch_size=8):
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
    for _, batch in tqdm(enumerate(loader), desc="Evaluating..."):
        img_emb: Tensor = batch["img_emb"]
        mask: Tensor = batch["mask"]
        previous_mask: Tensor = batch["previous_mask"]
        
        _, _, _, mask_input_torch = sam_train.prepare_prompt(
            original_size=original_size, mask_input=previous_mask
        )

        mask_pred, iou_pred, _ = sam_train.predict_torch(
            image_emb=img_emb,
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            mask_input=mask_input_torch,
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

        # if idx == 2:
        #     break

    # Each of these will eventually have shape
    # of [dataset_length, 3]
    return make_metrics(ious, dices, iou_mses, select_indices)

if __name__ == "__main__":
    dataset =FLARE22_SimpleMaskPropagate(
        metadata_path=VAL_METADATA,
        cache_name=FLARE22_SimpleMaskPropagate.VAL_CACHE_NAME,
        is_debug=False,
        device="cuda:0",
    )
    model: Sam = sam_model_registry["vit_b"](
            checkpoint="./sam_vit_b_01ec64.pth",
        )
    model.to("cuda:0")
    sam_train = SamTrain(sam_model=model)
    dataset.preload()
    metrics = evaluate(sam_train=sam_train, dataset=dataset)
    pp = PrettyPrinter(indent=2)
    with open(f"test-mask-prop.json", "w") as out:
        json.dump(metrics, out)
    pp.pprint(metrics)
    pass