from glob import glob
import os
from jinja2 import pass_eval_context
from matplotlib.widgets import EllipseSelector
import torch
from tqdm import tqdm
from scripts.datasets.constant import FLARE22_LABEL_ENUM, IMAGE_TYPE
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.utils import make_directory

preprocessor = FLARE22_Preprocess()

TRAIN_ROOT = "../dataset/FLARE22-version1/FLARE22_LabeledCase50"
TRAIN_IMAGE_PATH = os.path.join(TRAIN_ROOT, "images/*.nii.gz")
TRAIN_MASK_PATH = os.path.join(TRAIN_ROOT, "labels/*.nii.gz")

VAL_ROOT = "../dataset/FLARE22-version1/ReleaseValGT-20cases"
VAL_IMAGE_PATH = os.path.join(VAL_ROOT, "images/*.nii.gz")
VAL_MASK_PATH = os.path.join(VAL_ROOT, "labels/*.nii.gz")

train_images = sorted(list(glob(TRAIN_IMAGE_PATH)))
train_masks = sorted(list(glob(TRAIN_MASK_PATH)))
train_file = list(zip(train_images, train_masks))

val_images = sorted(list(glob(VAL_IMAGE_PATH)))
val_masks = sorted(list(glob(VAL_MASK_PATH)))
val_file = list(zip(val_images, val_masks))


class MeanShapeDataset(Dataset):
    TRAIN_CACHE_NAME = "mean-shape/train"
    VAL_CACHE_NAME = "mean-shape/validation"

    def __init__(
        self,
        mean_z: float = None,
        organ_idx: int = FLARE22_LABEL_ENUM.LIVER.value,
        is_train_set: bool = True,
    ) -> None:
        super().__init__()
        if is_train_set:
            self.files = train_file
            self.mean_z = None
        else:
            self.files = val_file
            assert (
                mean_z is not None
            ), f"Mean-z should be calculated by train-set and pass into validation set"
            self.mean_z = mean_z

        self.is_train_set = is_train_set
        self.organ_idx = organ_idx
        if self.is_train_set:
            self.cache_path = os.path.join(self.TRAIN_CACHE_NAME, "cache.pt")
        else:
            self.cache_path = os.path.join(self.VAL_CACHE_NAME, "cache.pt")
        make_directory(self.cache_path, is_file=True)
        if os.path.exists(self.cache_path):
            self.data = torch.load(self.cache_path)
            _, mean_z = self.load_file_and_extract_mean_z(files=self.files)
            self.mean_z = self.mean_z or mean_z
        else:
            self.preprocess()
            pass

    def preprocess(self):
        positive, negative = self.prepare_sdf(files=self.files)
        positive = torch.Tensor(positive)
        negative = torch.Tensor(negative)
        self.data = torch.concat([positive, negative], dim=0)
        torch.save(self.data, self.cache_path)
        pass

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]

    def load_file_and_extract_mean_z(self, files, is_reversed=True):
        masks = []
        for image_file, gt_file in tqdm(
            files, total=len(files), desc="Loading file..."
        ):
            _, mask_volumes = preprocessor.run_with_config(
                image_file=image_file,
                gt_file=gt_file,
                config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
            )
            if is_reversed:
                mask_volumes = mask_volumes[::-1]
            masks.append(mask_volumes)
            pass

        # Find the mean of z-axis
        mean_z = np.mean([mask.shape[0] for mask in masks])
        return masks, mean_z

    def prepare_sdf(self, files, organ_idx=FLARE22_LABEL_ENUM.LIVER.value):
        masks, mean_z = self.load_file_and_extract_mean_z(files)
        if self.mean_z is not None:
            mean_z = self.mean_z
        else:
            self.mean_z = mean_z

        masks = [mask == organ_idx for mask in masks]
        pos = []
        neg = []
        for mask in tqdm(masks, total=len(masks), desc="Calculate the SDF feature..."):
            positive = np.argwhere(mask) / np.array([[mean_z, 512, 512]])
            negative = np.argwhere(mask == False) / np.array([[mean_z, 512, 512]])
            positive = np.append(positive, np.ones((positive.shape[0], 1)), axis=-1)
            negative = np.append(negative, np.zeros((negative.shape[0], 1)), axis=-1)
            pos.append(positive)
            neg.append(negative)

        return positive, negative
