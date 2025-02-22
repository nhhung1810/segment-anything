from copy import deepcopy
import itertools
import os
import sys
from functools import reduce
from typing import Dict, List
from PIL import Image
import numpy as np
import glob
from pathlib import Path
from torch.nn.modules.module import _addindent
import torchvision.transforms as T
import torch

from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry


def resize(im: np.ndarray, target_size=[256, 256]):
    assert im.ndim <= 3, ""
    _im = torch.Tensor(im)
    _im = _im[None, ...] if im.ndim == 2 else _im

    [w, h] = target_size
    return T.Resize(size=(w, h))(_im)


def make_directory(path: str, is_file: bool = False):
    """Make nest directory from path.

    Args:
        path (str): target path
        is_file (bool, optional): if True, omit the file
            component and make directory. Defaults to False.
    """

    def make_nested_dir(directory: str) -> str:
        """Make nested Directory

        Args:
            directory (str): Path to directory

        Returns:
            str: Path to that directory
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        return directory

    if is_file:
        return make_nested_dir(os.path.dirname(path))

    return make_nested_dir(path)


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


def pick(d: Dict[object, object], keys: List[object]):
    return {k: v for k, v in d.items() if k in keys}


def omit(d: Dict[object, object], keys: List[object]):
    return {k: v for k, v in d.items() if k not in keys}


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "shape"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        if file is sys.stdout:
            main_str += ", \033[92m{:,}\033[0m params".format(total_params)
        else:
            main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, "w")
        print(string, file=file)
        file.flush()

    return count


def extract_non_image_encoder(
    model_path: str, save_path: str, model_type: str = "vit_b"
):
    if isinstance(torch.load(model_path), Sam):
        torch.save(torch.load(model_path).state_dict(), model_path)
        pass
    model: Sam = sam_model_registry[model_type](checkpoint=model_path)
    state_dict = deepcopy(model.state_dict())
    for key in list(state_dict.keys()):
        if not key.startswith("image_encoder."):
            continue
        del state_dict[key]
    torch.save(state_dict, save_path)
    pass


if __name__ == "__main__":
    files = list(glob.glob("./runs/sam_simple_obj_train-230426-095207/*pt"))
    for file in files:
        extract_non_image_encoder(file, file)
    pass


def torch_try_load(path: str, device: str, default_return={}) -> dict:
    try:
        return torch.load(path, map_location=device)
    except Exception as msg:
        pass
    return default_return
