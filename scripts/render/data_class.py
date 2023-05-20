from typing import Dict, List
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MaskData:
    mask: np.ndarray
    legend_dict: Dict[int, str]

    def __bool__(self):
        return self.mask is not None

    def format(self):
        if self.mask is None:
            return None

        assert self.mask.ndim <= 3, f"Mask dim <= 3, while get {self.mask.shape}"
        # Reshape mask to dim of 2
        if self.mask.ndim == 3:
            if self.mask.shape[0] == 1:
                mask = self.mask[0]
            elif self.mask.shape[-1] == 1:
                mask = self.mask[:, :, 0]
        else:
            mask = self.mask.copy()
        self.mask = mask

        return self


@dataclass
class MultiMasksData:
    masks: np.ndarray
    legend: List[str]

    def format(self):
        return self

    def __bool__(self):
        return self.masks is not None


@dataclass
class ImageData:
    image: np.ndarray

    def format(self):
        if self.image is None:
            return None

        # Check for img data
        assert self.image.ndim < 4, "Out of control"
        img = self.image.copy()
        if self.image.ndim == 2:
            img = img[:, :, None]

        if self.image.ndim == 3:
            assert (
                img.shape[-1] == 1 or img.shape[-1] == 3
            ), f"Invalid shape {self.image.shape} transform to {img.shape}"
            pass
        self.image = img
        return self

    def __bool__(self):
        return self.image is not None


@dataclass
class BBoxData:
    bbox: object

    def format(self):
        return self

    def __bool__(self):
        return self.bbox is not None


@dataclass
class PointData:
    points: object

    def format(self):
        return self

    def __bool__(self):
        return self.points is not None


@dataclass
class OneImageRenderData:
    image: Optional[ImageData] = None
    mask: Optional[MaskData] = None
    multi_masks: Optional[MultiMasksData] = None
    points: Optional[List[PointData]] = None
    bboxes: Optional[List[BBoxData]] = None

    def format(self):
        return self
