#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2022-05-12
@Author : nhhung1810
@File   : train.py

"""
from networkx import is_triad
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

import os
import sys
from copy import deepcopy
from datetime import datetime
from loguru import logger
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

# Internal
from scripts.datasets.constant import FLARE22_LABEL_ENUM, TRAIN_METADATA, VAL_METADATA
from scripts.datasets.flare22_mask_aug import FLARE22_MaskAug
from scripts.experiments.simple_mask_propagate.evaluate import evaluate
from scripts.experiments.ssm.dataset import MeanShapeDataset
from scripts.experiments.ssm.model import SimpleSDFModel
from scripts.losses import loss
from scripts.losses.loss import MultimaskSamLoss
from scripts.sam_train import SamTrain
from scripts.utils import summary
from tqdm import tqdm
from typing import Tuple

IS_DEBUG = False
NAME = "stat-shape"
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

    batch_size = 2048 * 4
    organ_idx = FLARE22_LABEL_ENUM.LIVER.value
    logdir = f"runs/{NAME}-{TIME}"
    n_epochs = 10
    save_epoch = 2
    evaluate_epoch = 1

    # Optim params
    learning_rate = 1e-4
    # learning_rate_decay_steps = 5
    learning_rate_decay_rate = 0.1
    learning_rate_decay_patience = 2

    ex.observers.append(FileStorageObserver.create(logdir))
    pass


@ex.capture
def make_dataset(batch_size, organ_idx) -> Tuple[MeanShapeDataset, DataLoader]:
    # Save GPU by host the dataset on cpu only
    dataset = MeanShapeDataset(is_train_set=True, organ_idx=organ_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataset, loader


@ex.capture
def make_model(
    device,
    learning_rate,
    learning_rate_decay_rate,
    learning_rate_decay_patience,
) -> Tuple[SimpleSDFModel, Optimizer, ReduceLROnPlateau, int]:
    model = SimpleSDFModel(n_input=3, n_output=1, n_hidden=10)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    logger.info("Model structure")
    summary(model)

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=learning_rate_decay_rate,
        patience=learning_rate_decay_patience,
        verbose=True,
    )

    loss_fnc = torch.nn.BCELoss()
    loss_fnc.to(device=device)

    return model, optimizer, scheduler, loss_fnc


@ex.capture
def run_evaluate(
    model: SimpleSDFModel, device: str, organ_idx: int, mean_z: float
) -> dict:
    dataset = MeanShapeDataset(mean_z=mean_z, is_train_set=False, organ_idx=organ_idx)
    loss_fnc = torch.nn.BCELoss()
    metrics = {}
    with torch.no_grad():
        pred = model(dataset.data[:, :3])
        label = dataset.data[:, 3:]
        val_loss: Tensor = loss_fnc(pred, label)
        p, r, f1, o = precision_recall_fscore_support(
            y_true=label.cpu().numpy(), y_pred=pred.cpu().numpy()
        )
        metrics["loss"] = val_loss.item()
        metrics["precision"] = p
        metrics["recall"] = r
        metrics["f1"] = f1

    return metrics


@ex.automain
def train(
    logdir,
    device,
    n_epochs,
    save_epoch,
    evaluate_epoch,
):
    print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_dataset, loader = make_dataset()
    model, optimizer, scheduler, loss_fnc = make_model()

    assert isinstance(model, SimpleSDFModel), ""
    assert isinstance(train_dataset, MeanShapeDataset), ""
    assert isinstance(optimizer, Optimizer), ""
    assert isinstance(scheduler, ReduceLROnPlateau), ""
    assert isinstance(loss_fnc, torch.nn.BCELoss), ""

    optimizer.zero_grad()
    loop = tqdm(range(1, n_epochs + 1), total=n_epochs, desc="Training...")
    for batch_idx in loop:
        one_batch_losses = []
        for idx, batch in tqdm(
            enumerate(loader), desc=f"Epoch {batch_idx}", leave=False
        ):
            # 1 mask have to clone to fit the number of prompt
            # and the number of multiple-mask

            feature = batch[:, :3]
            label = batch[:, 3:]
            pred = model(feature)
            loss = loss_fnc(pred, label)

            # Optimize
            loss.backward()
            one_batch_losses.append(loss.detach().cpu().numpy())
            optimizer.step()
            optimizer.zero_grad()
            pass

        # Loss by batch
        writer.add_scalar(
            "train/loss", np.array(one_batch_losses).mean(), global_step=batch_idx
        )

        writer.add_scalar(
            "train/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step=batch_idx,
        )

        if batch_idx % evaluate_epoch == 0:
            metrics = run_evaluate(model=model, mean_z=train_dataset.mean_z)
            for k, v in metrics.items():
                writer.add_scalar(f"validation/{k}", v, global_step=batch_idx)

            scheduler.step(metrics["loss"])

            pass

        if batch_idx % save_epoch == 0:
            model_path = os.path.join(logdir, f"model-{batch_idx}.pt")
            torch.save(model.state_dict(), model_path)
            pass
