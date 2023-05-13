#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2022-05-12
@Author : nhhung1810
@File   : train.py

"""
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam

import os
import sys
from copy import deepcopy
from datetime import datetime
from loguru import logger
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from scripts.datasets.constant import FLARE22_LABEL_ENUM, TRAIN_METADATA, VAL_METADATA

# Internal
from scripts.datasets.flare22_simple_mask_propagate import FLARE22_SimpleMaskPropagate
from scripts.experiments.simple_mask_propagate.evaluate import evaluate
from scripts.losses.loss import MultimaskSamLoss
from scripts.sam_train import SamTrain
from scripts.utils import summary
from tqdm import tqdm
from typing import Tuple

IS_DEBUG = False
NAME = "mp-focus"
TIME = datetime.now().strftime("%y%m%d-%H%M%S")
ex = Experiment(NAME)

# NOTE: Refactoring
logger.remove()
logger.add(
    sys.stdout,
    format="<lvl>[{time:DD:MMM:YY HH:mm:ss}] - [{level}] - {message}</lvl>",
)


# Reproducibility

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


@ex.config
def config():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 16

    logdir = f"runs/{NAME}-{TIME}"
    custom_model_path = None
    class_selected = [
        FLARE22_LABEL_ENUM.LIVER.value,
        FLARE22_LABEL_ENUM.GALLBLADDER.value,
        FLARE22_LABEL_ENUM.IVC.value,
        FLARE22_LABEL_ENUM.RIGHT_KIDNEY.value
    ]
    n_epochs = 100
    save_epoch = 5
    evaluate_epoch = 10

    gradient_accumulation_step = 4

    # Model params
    focal_gamma = 2.0
    focal_alpha = None

    # Optim params
    learning_rate = 6e-6
    learning_rate_decay_steps = 5
    learning_rate_decay_rate = 0.98

    ex.observers.append(FileStorageObserver.create(logdir))
    pass


@ex.capture
def make_dataset(
    device, batch_size, class_selected
) -> Tuple[FLARE22_SimpleMaskPropagate, DataLoader]:
    # Save GPU by host the dataset on cpu only
    dataset = FLARE22_SimpleMaskPropagate(
        metadata_path=TRAIN_METADATA,
        cache_name=FLARE22_SimpleMaskPropagate.TRAIN_CACHE_NAME,
        is_debug=IS_DEBUG,
        device=device,
        class_selected=class_selected,
    )
    dataset.preprocess()
    dataset.preload()
    dataset.self_check()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Make sure that the evaluation dataset also work
    FLARE22_SimpleMaskPropagate(
        metadata_path=VAL_METADATA,
        cache_name=FLARE22_SimpleMaskPropagate.VAL_CACHE_NAME,
        is_debug=IS_DEBUG,
        device="cpu",
        class_selected=class_selected,
    ).preprocess()
    return dataset, loader


@ex.capture
def make_model(
    device,
    custom_model_path,
    learning_rate,
    learning_rate_decay_steps,
    learning_rate_decay_rate,
    focal_gamma,
    focal_alpha,
) -> Tuple[SamTrain, Optimizer, StepLR, int]:
    def load_model(
        checkpoint="./sam_vit_b_01ec64.pth",
        checkpoint_type="vit_b",
        custom_model_path: str = None,
    ) -> Sam:
        sam: Sam = sam_model_registry[checkpoint_type](
            checkpoint=checkpoint, custom=custom_model_path
        )
        return sam

    model = load_model(custom_model_path=custom_model_path)
    model.to(device=device)

    if custom_model_path is None:
        logger.warning("Initialize from pre-train")
    else:
        logger.success("Load custom model")
        pass

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    logger.info("Model sctructure")
    summary(model.prompt_encoder)
    summary(model.mask_decoder)

    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate
    )

    sam_train = SamTrain(sam_model=model)

    loss_fnc = MultimaskSamLoss(
        reduction="mean", focal_alpha=focal_alpha, focal_gamma=focal_gamma
    )

    loss_fnc.to(device=device)

    return sam_train, optimizer, scheduler, loss_fnc


def checkpoint(model: Sam, device: str, save_path: str):
    model.to("cpu")
    state_dict = deepcopy(model.state_dict())
    # Remove image_encoder for lighter weight
    for key in list(state_dict.keys()):
        if not key.startswith("image_encoder."):
            continue
        del state_dict[key]
    torch.save(state_dict, save_path)
    model.to(device)

    pass


@ex.capture
def run_evaluate(sam_train: SamTrain, device: str, class_selected) -> dict:
    dataset = FLARE22_SimpleMaskPropagate(
        metadata_path=VAL_METADATA,
        cache_name=FLARE22_SimpleMaskPropagate.VAL_CACHE_NAME,
        is_debug=IS_DEBUG,
        device=device,
        class_selected=class_selected,
    )
    dataset.preload()
    return evaluate(sam_train, dataset)


@ex.automain
def train(
    logdir,
    device,
    n_epochs,
    save_epoch,
    evaluate_epoch,
    gradient_accumulation_step,
):
    print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_dataset, loader = make_dataset()
    sam_train, optimizer, scheduler, loss_fnc = make_model()

    assert isinstance(sam_train, SamTrain), ""
    assert isinstance(train_dataset, FLARE22_SimpleMaskPropagate), ""
    assert isinstance(optimizer, Optimizer), ""
    assert isinstance(scheduler, StepLR), ""
    assert isinstance(loss_fnc, MultimaskSamLoss), ""

    optimizer.zero_grad()
    sam_train.model.train()
    loop = tqdm(range(1, n_epochs + 1), total=n_epochs, desc="Training...")
    input_size, original_size = train_dataset.get_size()
    for batch_idx in loop:
        one_batch_losses = []
        for idx, batch in tqdm(
            enumerate(loader), desc=f"Epoch {batch_idx}", leave=False
        ):
            img_emb: Tensor = batch["img_emb"]
            mask: Tensor = batch["mask"]
            previous_mask: Tensor = batch["previous_mask"]

            _, _, _, mask_input_torch = sam_train.prepare_prompt(
                original_size=original_size, mask_input=previous_mask
            )

            masks_pred, iou_pred, _ = sam_train.predict_torch(
                image_emb=img_emb,
                input_size=input_size,
                original_size=original_size,
                mask_input=mask_input_torch,
                multimask_output=True,
                return_logits=True,
            )

            # 1 mask have to clone to fit the number of prompt
            # and the number of multiple-mask
            mask = mask.unsqueeze(1).repeat_interleave(3, dim=1).type(torch.int64)

            loss = loss_fnc.forward(
                multi_mask_pred=masks_pred,
                multi_iou_pred=iou_pred,
                multi_mask_target=mask,
            )

            # Before doing anything, we have to record the actual loss
            one_batch_losses.append(loss.detach().cpu().numpy())
            # Normalize loss
            loss: Tensor = loss / gradient_accumulation_step
            loss.backward()

            # This will update the gradient at once
            if idx % gradient_accumulation_step == 0:
                optimizer.step()
                # NOTE: clear the gradient
                optimizer.zero_grad()

            pass

        # Loss by batch
        writer.add_scalar(
            "train/loss", np.array(one_batch_losses).mean(), global_step=batch_idx
        )
        writer.add_scalar(
            "train/learning_rate", scheduler.get_last_lr()[0], global_step=batch_idx
        )

        # End 1 epoch
        # LR-scheduler run by epoch
        scheduler.step()
        pass

        # The idx from the above loop will be leak out of scope, this is python feat.
        if idx % gradient_accumulation_step != 0:
            # Run backward if there some gradient left
            optimizer.step()
            optimizer.zero_grad()
            pass

        if batch_idx % evaluate_epoch == 0:
            # offload_gpu(dataset=train_dataset)
            metrics: dict = run_evaluate(sam_train=sam_train)
            for k, v in metrics.items():
                writer.add_scalar(f"validation/{k}", v, global_step=batch_idx)
            pass

        if batch_idx % save_epoch == 0:
            model_path = os.path.join(logdir, f"model-{batch_idx}.pt")
            checkpoint(sam_train.model, device, model_path)
            pass
