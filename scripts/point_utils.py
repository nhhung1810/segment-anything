from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
import numpy as np
from utils import argmax_dist


@dataclass
class BBoxProperty:
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    to_value: object
    pass


def mask_out(mask, bbox: BBoxProperty):
    _mask = np.ones(mask.shape) == 1.0
    _mask[bbox.xmin : bbox.xmax, bbox, bbox.ymin : bbox.ymax] = False
    mask[_mask] = bbox.to_value
    return mask


class PointUtils:
    def positive_center_point(self, mask, class_number):
        filter_center_candidate = gaussian_filter(mask == class_number, 10)
        #  Precise positive
        [row, col] = np.argwhere(filter_center_candidate > 0.0)[0]
        return row, col

    def positive_random_point(self, mask, class_number, center: list = None):
        [row, col] = center
        positive = np.argwhere((mask == class_number).astype(np.int16) > 0.0)
        choices = np.random.RandomState(seed=1810).choice(
            np.arange(positive.shape[0]), size=10
        )
        c = argmax_dist(positive[choices], row, col)
        positive = positive[c : c + 1][:, ::-1]
        return positive

    def negative_random_outside(self, mask):
        filter_negative = gaussian_filter(mask == 0, 3)
        filter_negative = mask_out(
            filter_negative, BBoxProperty(200, 450, 100, 450, 0.0)
        )
        negative = np.argwhere(filter_negative > 0.0)
        choices = np.random.RandomState(seed=1810).choice(
            np.arange(negative.shape[0]), size=1
        )
        # Pickup the choice and swap row with col
        negative = negative[choices][:, ::-1]
        return negative

    # Use cases:

    def one_positive(self, mask, class_number):
        coors = np.argwhere(mask == class_number)

        if coors.shape[0] == 0:
            return None, None

        row, col = self.positive_center_point(mask, class_number)

        # Make the col/row and label
        coors = np.array([[col, row]])
        label = np.array([1])

        # In Cartesian Coordinate, number of row is y-axis
        return coors, label

    def two_positive_one_negative(self, mask, class_number):
        coors = np.argwhere(mask == class_number)

        if coors.shape[0] == 0:
            return None, None

        row, col = self.positive_center_point(mask, class_number)
        positive = self.positive_random_point(mask, class_number, [row, col])
        negative = self.negative_random(mask)

        # Make the col/row and label
        coors = np.array([[col, row], *positive, *negative])
        label = np.array([1, *np.ones(positive.shape[0]), *np.zeros(negative.shape[0])])

        # In Cartesian Coordinate, number of row is y-axis
        return coors, label

    def no_prompt(self):
        return None, None
