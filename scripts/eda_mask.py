from collections import defaultdict
import json
import os
from pydoc import pathdirs
import subprocess
from typing import List
import natsort
import numpy as np

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from tqdm import tqdm


from scripts.datasets.constant import (
    IMAGE_TYPE,
    TRAIN_NON_PROCESSED,
    VAL_METADATA,
    FLARE22_LABEL_ENUM,
)
from scripts.datasets.flare22_loader import FileLoader
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.experiments.simple_mask_propagate.inference import find_organ_range
from scripts.render.render import Renderer
from scripts.utils import load_file_npz, load_img, make_directory, omit
from segment_anything.utils.transforms import ResizeLongestSide


PATH = "/dataset/FLARE22-version1"
VAL_MASK = f"{PATH}/ValMask"
VAL_IMAGE = f"{PATH}/ValImageProcessed"


def get_all_organ_range(masks):
    starts, ends = [], []
    for class_num in range(1, 14, 1):
        start_idx, end_idx = find_organ_range(masks=masks, class_num=class_num)
        starts.append(start_idx)
        ends.append(end_idx)

    # Padding zero for background pixel
    return np.array([0, *starts], np.int32), np.array([0, *ends], np.int32)


def resize_to(img: np.ndarray, target_size):
    return np.array(resize(to_pil_image(img), target_size))


def calculate_bbox(mask):
    bboxes = [(dict(x=0, y=0, w=0, h=0), 0)]
    for class_num in range(1, 14, 1):
        coors = np.argwhere(mask == class_num)
        if not coors.shape[0] == 0:
            xs = coors[:, 0]
            ys = coors[:, 1]
            top_left = [xs.min(), ys.min()]
            right_bottom = [xs.max(), ys.max()]
            w = top_left[0] - right_bottom[0]
            h = top_left[1] - right_bottom[1]
            # Invert the data for correct orientation
            bboxes.append(
                (dict(x=right_bottom[1], y=right_bottom[0], w=h, h=w), class_num)
            )
        else:
            bboxes.append((dict(x=0, y=0, w=0, h=0), class_num))
    return bboxes


def visualize(
    images: List[str], gts: List[str], direction: str = "T", framerate: int = 10
):
    preprocessor = FLARE22_Preprocess()
    # render = Renderer(
    #     legend_dict={
    #         value.value: value.name.lower().replace("_", " ")
    #         for value in FLARE22_LABEL_ENUM
    #     }
    # )
    all_area_data = {}
    all_point_data = {}
    for image_file, gt_file in tqdm(
        zip(images, gts), total=len(images), desc="EDA for patient..."
    ):
        patient_name = os.path.basename(image_file).replace(".nii.gz", "")
        area_data = defaultdict(list)

        # shape = [T, H, W]
        volumes, masks = preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )
        if direction == "T":
            pass
        elif direction == "W":
            # T, H, W = volumes.shape
            volumes = volumes.transpose(2, 0, 1)
            masks = masks.transpose(2, 0, 1)
            pass

        starts, ends = get_all_organ_range(masks)
        all_point_data[patient_name] = {
            class_num: [
                float(starts[class_num] / volumes.shape[0]),
                float(ends[class_num] / volumes.shape[0]),
            ]
            for class_num in range(1, 14)
        }
        for idx in tqdm(
            range(volumes.shape[0]),
            desc="Frame render...",
            total=volumes.shape[0],
            leave=False,
        ):
            img = volumes[idx, ...]
            mask = masks[idx, ...]

            # Calculate the area
            image_size = img.shape[0] * img.shape[1]
            for class_num in range(1, 14):
                area_ratio = np.sum((mask == class_num).astype(np.float32)) / image_size
                area_data[class_num].append(float(area_ratio))

            pass

        all_area_data[patient_name] = area_data
        # break

    with open(f"all_area_data-{direction}.json", "w") as out:
        json.dump(all_area_data, out)

    with open(f"all_point_data-{direction}.json", "w") as out:
        json.dump(all_point_data, out)

    pass


if __name__ == "__main__":
    image_dir = f"{TRAIN_NON_PROCESSED}/images"
    label_dir = f"{TRAIN_NON_PROCESSED}/labels"
    images_path: List[str] = sorted(os.listdir(image_dir))
    labels_path = [
        os.path.join(label_dir, p.replace("_0000.nii.gz", ".nii.gz"))
        for p in images_path
    ]
    images_path = [os.path.join(image_dir, p) for p in images_path]

    framerate = 10
    direction = "W"
    visualize(
        images=images_path, gts=labels_path, framerate=framerate, direction=direction
    )
