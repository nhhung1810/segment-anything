# This config get from IVOS project of anh Thang Long and his friend
# https://github.com/kaylode/ivos/tree/master/tools/preprocess/windowing_ct/config.py
from typing import List
import nibabel as nib

import numpy as np

from scripts.datasets.constant import FLARE22_LABEL_ENUM, IMAGE_TYPE
from scripts.tools.evaluation.loading import change_axes_of_image, load_ct_info
# from scripts.datasets.transform import TRANSFORM


WINDOW_CT_CONFIG = {
    "spine-bone": [{"W": 1800, "L": 400}],
    "abdomen-soft_tissues_abdomen-liver": [
        {"W": 400, "L": 50},
        {"W": 150, "L": 30},
    ],
    "chest-lungs_chest-mediastinum": [{"W": 1500, "L": -600}, {"W": 350, "L": 50}],
}


"""
What we want: input a file.nii.gz as image input, 
specify the config and we get the output as processed 3D
np.ndarray of shape [N, H, W], with a save to disk option.
"""


class FLARE22_Preprocess:
    def __init__(self) -> None:
        pass

    def run_with_config(self, image_file: str, gt_file: str, config_name: IMAGE_TYPE):
        assert config_name in IMAGE_TYPE
        assert ".nii.gz" in image_file
        assert ".nii.gz" in gt_file
        c = WINDOW_CT_CONFIG[config_name.value.replace(" ", "_")]

        # Load the CT images
        image_dict = load_ct_info(image_file)
        mask_dict = load_ct_info(gt_file)

        # Rotate the image and label
        sub_direction = image_dict["subdirection"]
        image_dict["npy_image"] = change_axes_of_image(
            image_dict["npy_image"], sub_direction
        )
        mask_dict["npy_image"] = change_axes_of_image(
            mask_dict["npy_image"], sub_direction
        )

        # Preprocess images with correct window config
        preprocessed_volume = self.run(
            imgs=image_dict["npy_image"],
            window_level=[x["L"] for x in c],
            window_width=[x["W"] for x in c],
        )

        return preprocessed_volume, mask_dict["npy_image"]

    def run(
        self,
        imgs: np.ndarray,
        window_level: List[int],
        window_width: List[int],
    ) -> np.ndarray:
        window_min, window_max = self.compute_window(window_level, window_width)
        processed_volume = []
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :]
            img = np.clip(img, window_min, window_max)
            img = 255 * ((img - window_min) / (window_max - window_min))
            img = img.astype(np.uint8)
            processed_volume.append(img)

        processed_volume = np.stack(processed_volume, axis=0)
        return processed_volume

    def compute_window(self, window_level: List[int], window_width: List[int]):
        window_min, window_max = None, None
        for level, width in zip(window_level, window_width):
            window_min = (
                level - (width // 2)
                if window_min is None
                else min(window_min, level - (width // 2))
            )
            window_max = (
                level + (width // 2)
                if window_max is None
                else max(window_max, level + (width // 2))
            )
        return window_min, window_max

    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    processor = FLARE22_Preprocess()

    volume, mask = processor.run_with_config(
        image_file="./dataset/FLARE22-version1/FLARE22_LabeledCase50/images/FLARE22_Tr_0008_0000.nii.gz",
        gt_file="./dataset/FLARE22-version1/FLARE22_LabeledCase50/labels/FLARE22_Tr_0008.nii.gz",
        config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
    )
    print("🚀 ~ file: preprocess_raw.py:91 ~ volume:", volume.shape)
    print("🚀 ~ file: preprocess_raw.py:91 ~ mask:", mask.shape)
    v = volume[61][..., None].repeat(3, axis=-1)
    m = mask[61]
    f, axes = plt.subplots(1, 2)
    axes[0].imshow(v)
    axes[1].imshow(m)
    plt.show()

    pass
