import itertools
from PIL import Image
import numpy as np
import glob
from pathlib import Path

import torchvision.transforms as T

import torch

def resize(im : np.ndarray, target_size = [256, 256]):
    assert im.ndim <= 3, ''
    _im = torch.Tensor(im)
    _im = _im[None, ...] if im.ndim == 2 else _im
    
    [w, h] = target_size
    return T.Resize(size = (w, h))(_im)

def make_nested_dir(directory: str) -> str:
    """Make nested Directory

    Args:
        directory (str): Path to directory

    Returns:
        str: Path to that directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


GROUP1 = "FLARE22_Tr_0001_0000_abdomen-soft tissues_abdomen-liver"
GROUP2 = "FLARE22_Tr_0001_0000_chest-lungs_chest-mediastinum"
GROUP3 = "FLARE22_Tr_0001_0000_spine-bone"

def get_data_paths(GROUP):
    data = list(glob.glob(f"../dataset/FLARE-small/{GROUP}/*"))
    mask = list(glob.glob("../dataset/FLARE-small/FLARE22_Tr_0001_0000-mask/*"))
    data = sorted(data)
    mask = sorted(mask)
    return data, mask

def load_img(path):
    return np.asarray(Image.open(path).convert("RGB"))

def load_file_npz(npz_path) -> np.ndarray:
    return np.load(npz_path)

def argmax_dist(coors, x, y):
    return np.argmax(np.sqrt(np.square(coors[:, 0] - x) + np.square(coors[:, 0] - y)))

def generate_grid(w, h, est_n_point=16):
    n_axis_sampling = int(np.sqrt(est_n_point))
    w_axis = np.linspace(0, w, n_axis_sampling)
    h_axis = np.linspace(0, h, n_axis_sampling)
    samples = np.array(list(itertools.product(w_axis, h_axis))).astype(np.int32)
    labels = np.ones(n_axis_sampling**2)
    return samples, labels

def mask_out(mask, xmin, xmax, ymin, ymax, to_value):
    _mask = np.ones(mask.shape) == 1.0
    _mask[xmin:xmax, ymin:ymax] = False
    mask[_mask] = to_value
    return mask