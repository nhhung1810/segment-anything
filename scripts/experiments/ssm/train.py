#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2022-05-12
@Author : nhhung1810
@File   : train.py

"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

import os
import sys
from datetime import datetime
from loguru import logger
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

# Internal
from scripts.datasets.constant import FLARE22_LABEL_ENUM
from scripts.experiments.ssm.dataset import MeanShapeDataset
from scripts.experiments.ssm.model import SimpleSDFModel
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

    batch_size = 2048 * 128
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
def make_dataset(batch_size, organ_idx, device) -> Tuple[MeanShapeDataset, DataLoader]:
    # Save GPU by host the dataset on cpu only
    train_dataset = MeanShapeDataset(is_train_set=True, organ_idx=organ_idx)
    val_dataset = MeanShapeDataset(is_train_set=False, organ_idx=organ_idx, mean_z=train_dataset.mean_z)
    if batch_size is None:
        batch_size = len(train_dataset)
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataset, val_dataset, loader


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
    model: SimpleSDFModel, device: str, dataset: MeanShapeDataset, batch_size: int,
) -> dict:
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    loss_fnc = torch.nn.BCELoss()
    loss_fnc.to(device)
    metrics = {}
    losses = []
    labels = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch[:, :3])
            label = batch[:, 3:]
            val_loss: Tensor = loss_fnc(pred, label)
            losses.append(val_loss.cpu().item())
            labels.append(label.cpu().numpy())
            preds.append(pred.cpu().numpy())
        pass

    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    p, r, f1, o = precision_recall_fscore_support(
        y_true=labels, y_pred=preds
    )
    val_loss = np.mean(losses)
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

    train_dataset, val_dataset, loader = make_dataset()
    model, optimizer, scheduler, loss_fnc = make_model()

    assert isinstance(model, SimpleSDFModel), ""
    assert isinstance(train_dataset, MeanShapeDataset), ""
    assert isinstance(val_dataset, MeanShapeDataset), ""
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
            batch = batch.to(device)
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
            metrics = run_evaluate(model=model, dataset=val_dataset)
            for k, v in metrics.items():
                writer.add_scalar(f"validation/{k}", v, global_step=batch_idx)

            scheduler.step(metrics["loss"])

            pass

        if batch_idx % save_epoch == 0:
            model_path = os.path.join(logdir, f"model-{batch_idx}.pt")
            torch.save(model.state_dict(), model_path)
            pass
