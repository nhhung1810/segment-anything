import torch
from enum import Enum

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class FLARE22_LABEL_ENUM(Enum):
    BACK_GROUND = 0
    LIVER = 1
    RIGHT_KIDNEY = 2
    SPLEEN = 3
    PANCREAS = 4
    AORTA = 5
    IVC = 6  # Inferior Vena Cava
    RAG = 7  # Right Adrenal Gland,
    LAG = 8  # Left Adrenal Gland,
    GALLBLADDER = 9
    ESOPHAGUS = 10
    STOMACH = 11
    DUODENUM = 12
    LEFT_KIDNEY = 13


class IMAGE_TYPE(Enum):
    ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER = "abdomen-soft tissues_abdomen-liver"
    CHEST_LUNGS_CHEST_MEDIASTINUM = "chest-lungs_chest-mediastinum"
    SPINE_BONE = "spine-bone"


# Respective to root
DATASET_ROOT = "./dataset/FLARE22-version1/"
TRAIN_NON_PROCESSED = "./dataset/FLARE22-version1/FLARE22_LabeledCase50"

# Processed data
TRAIN_PATH = "./dataset/FLARE22-version1/TrainImageProcessed"
TRAIN_MASK = "./dataset/FLARE22-version1/TrainMask"
TRAIN_METADATA = "./dataset/FLARE22-version1/train_metadata.json"
VAL_METADATA = "./dataset/FLARE22-version1/val_metadata.json"
