import json

from tqdm import tqdm
import numpy as np
import os
import torch
from torch import Tensor
from scripts.datasets.constant import (
    DATASET_ROOT,
    IMAGE_TYPE,
    FLARE22_LABEL_ENUM,
    TRAIN_METADATA,
    VAL_METADATA,
)
import albumentations as A
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from scripts.sam_train import SamTrain
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry
from scripts.utils import (
    load_file_npz,
    load_img,
    make_directory,
    omit,
    pick,
)


class FLAREElement:
    patient: str
    patient_masks: List[str]
    patient_image_group: List[str]
    class_list: List[IMAGE_TYPE]
    chest_lungs_chest_mediastinum: Optional[List[str]]
    abdomen_soft_tissues_abdomen_liver: Optional[List[str]]
    spine_bone: Optional[List[str]]

    def __init__(
        self,
        patient: str = None,
        patient_masks: List[str] = None,
        patient_image_group: List[str] = None,
        class_list: List[IMAGE_TYPE] = None,
        chest_lungs_chest_mediastinum: Optional[List[str]] = None,
        abdomen_soft_tissues_abdomen_liver: Optional[List[str]] = None,
        spine_bone: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.patient = patient
        self.patient_image_group = patient_image_group
        self.class_list = class_list
        self.patient_masks = sorted(patient_masks)
        self.chest_lungs_chest_mediastinum = sorted(chest_lungs_chest_mediastinum)
        self.abdomen_soft_tissues_abdomen_liver = sorted(
            abdomen_soft_tissues_abdomen_liver
        )
        self.spine_bone = sorted(spine_bone)

    def generate_path(self, dataset_root):
        for img_path, mask_path in zip(
            self.abdomen_soft_tissues_abdomen_liver, self.patient_masks
        ):
            id_number = int(os.path.splitext(mask_path)[0].rsplit("_", 1)[-1])
            yield {
                "name": self.patient,
                "id_number": id_number,
                "img_path": os.path.join(dataset_root, img_path),
                "mask_path": os.path.join(dataset_root, mask_path),
            }
            pass
        pass


class FileLoader:
    def __init__(
        self,
        metadata_path: str = TRAIN_METADATA,
        dataset_root: str = DATASET_ROOT,
    ) -> None:
        with open(metadata_path) as out:
            self.metadata = json.load(out)
        self.data: List[FLAREElement] = []
        self.dataset_root = dataset_root
        for data in self.metadata:
            if (
                len(data.get(IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER.value, []))
                == 0
            ):
                continue
            e = FLAREElement(
                **{k.replace("-", "_").replace(" ", "_"): v for k, v in data.items()}
            )
            self.data.extend(e.generate_path(dataset_root))
            pass
        pass


class FLARE22_MaskDrop(Dataset):
    # FOR DEBUG
    LIMIT = 20
    TRAIN_CACHE_NAME = "flare22-mask-drop/train"
    VAL_CACHE_NAME = "flare22-mask-drop/validation"

    def __init__(
        self,
        pre_trained_sam: Sam = None,
        metadata_path: str = TRAIN_METADATA,
        dataset_root: str = DATASET_ROOT,
        cache_name: str = TRAIN_CACHE_NAME,
        is_debug: bool = False,
        device: str = "cpu",
        class_selected: List[int] = None,
        allow_augmentation: List[int] = [],
        augmentation_prop: float = 0.5,
    ) -> None:
        super().__init__()
        # For preprocess data
        self.device = device
        self.class_selected = class_selected or list(range(1, 14))
        self.pre_trained_sam = pre_trained_sam
        self.dataset_root = dataset_root
        self.allow_augmentation = allow_augmentation
        self.augmentation_prop = augmentation_prop

        if pre_trained_sam:
            self.pre_trained_sam.eval()
            self.train_sam = SamTrain(self.pre_trained_sam)

        self.file = FileLoader(metadata_path=metadata_path, dataset_root=dataset_root)

        if is_debug:
            self.file.data = self.file.data[: self.LIMIT]
        self.cache_path = os.path.join(self.dataset_root, cache_name)
        make_directory(self.cache_path)
        self.dataset = []

        self.input_size = None
        self.original_size = None

    def __getitem__(self, index) -> Dict:
        data = self.dataset[index]
        return self.format_batch(data)

    def format_batch(self, data: dict):
        """Format into batch-compatible data

        Args:
            data (dict): Data from preprocessing step

        Returns:
            Dict: batch-compatible data for dataloader
        """

        # Remove the batch element
        img_emb = data["img_emb"][0]
        mask = data["mask"]
        previous_mask = data["previous_mask"]
        class_number = data["class_number"]
        if class_number in self.allow_augmentation:
            previous_mask = self.drop_mask(previous_mask)
            pass
        return dict(
            img_emb=img_emb.to(self.device),
            mask=mask.to(self.device),
            previous_mask=previous_mask.to(self.device),
        )

    def mask_drop(mask: Tensor, class_num: int, max_crop_ratio=0.5):
        coors = np.argwhere(mask == class_num)
        if coors.shape[0] == 0:
            return mask
        xs = coors[:, 1]
        ys = coors[:, 0]
        top_left = [xs.min(), ys.min()]
        right_bottom = [xs.max(), ys.max()]
        w, h = [right_bottom[0] - top_left[0], right_bottom[1] - top_left[1]]

        # Define the max_crop
        max_crop = int(min(w, h) * max_crop_ratio)

        drop_fn = A.CoarseDropout(
            max_holes=1,
            max_height=int(max_crop),
            max_width=int(max_crop),
            fill_value=0.0,
            always_apply=True,
        )
        y_slice = slice(top_left[1], right_bottom[1])
        x_slice = slice(top_left[0], right_bottom[0])
        sub_mask = mask[y_slice, x_slice]
        sub_mask = drop_fn(image=sub_mask)["image"]
        mask[y_slice, x_slice] = sub_mask
        return mask

    def drop_mask(self, previous_mask: Tensor):
        coors = torch.argwhere(previous_mask)
        if coors.shape[0] == 0:
            return previous_mask
        xs = coors[:, 1]
        ys = coors[:, 0]
        top_left = [xs.min(), ys.min()]
        right_bottom = [xs.max(), ys.max()]
        w, h = [right_bottom[0] - top_left[0], right_bottom[1] - top_left[1]]
        max_crop = int(min(w, h) * self.max_crop_ratio)
        drop_fn = A.CoarseDropout(
            max_holes=1,
            max_height=int(max_crop),
            max_width=int(max_crop),
            fill_value=0.0,
            p=self.prob_augmentation,
        )
        # Sub-region augmentation only
        y_slice = slice(top_left[1], right_bottom[1])
        x_slice = slice(top_left[0], right_bottom[0])
        sub_mask = previous_mask[y_slice, x_slice]
        sub_mask = drop_fn(image=sub_mask)["image"]
        previous_mask[y_slice, x_slice] = sub_mask
        return previous_mask

    def __len__(self):
        return len(self.dataset)

    def get_size(self):
        input_size = (self.input_size[0], self.input_size[1])
        original_size = (self.original_size[0], self.original_size[1])
        return input_size, original_size

    def preload(self):
        if len(self.dataset) > 0:
            return
        for data in tqdm(
            self.file.data, desc="Preload data to RAM", total=len(self.file.data)
        ):
            current_id = int(data["id_number"])
            previous_id = current_id - 1

            current_cache_path = os.path.join(
                self.cache_path,
                f"{data['name']}/{current_id}.pt",
            )
            previous_cache_path = os.path.join(
                self.cache_path, f"{data['name']}/{previous_id}.pt"
            )

            if not os.path.exists(current_cache_path):
                continue
            if not os.path.exists(previous_cache_path):
                continue

            current_data_dict = torch.load(current_cache_path)
            previous_data_dict = torch.load(previous_cache_path)

            self.load_data_dict_into_dataset(current_data_dict, previous_data_dict)

        pass

    def preprocess(self):
        """Efficient caching and broadcast cache into pair of embedding-mask"""
        if len(self.dataset) != 0:
            return
        for data in tqdm(
            self.file.data,
            desc="Preprocessing/Checking embedding",
            total=len(self.file.data),
        ):
            cache_path = os.path.join(
                self.cache_path, f"{data['name']}", f"{data['id_number']}.pt"
            )
            # Skip when cache-data existed
            if os.path.exists(cache_path):
                continue

            make_directory(cache_path, is_file=True)
            masks = load_file_npz(data["mask_path"])
            # All data, sometime it has invalid mask for a target -> discard it
            if self.is_not_valid_mask(masks):
                continue
            # Calculate the embedding
            img = load_img(data["img_path"])
            assert self.train_sam is not None, "Preprocess require the train-sam module"
            img_emb, original_size, input_size = self.train_sam.prepare_img(image=img)

            # Compose the cache
            data_dict = self.compose_cache(masks, img_emb, original_size, input_size)
            # Cache embedding and mask
            torch.save(data_dict, cache_path)
        pass

    def compose_cache(
        self, masks: np.ndarray, img_emb: Tensor, original_size, input_size
    ):
        data_dict = {
            "img_emb": img_emb,
            "original_size": torch.as_tensor(original_size),
            "input_size": torch.as_tensor(input_size),
        }

        for class_value in np.unique(masks):
            if class_value == 0:
                # Skip class 0
                continue

            _value = {
                "mask": torch.as_tensor(masks == class_value),
            }
            data_dict[f"{class_value}"] = _value
            pass

        data_dict["n_masks"] = np.unique(masks).shape[0]
        return data_dict

    def is_not_valid_mask(self, masks):
        """Filter out bad mask"""
        return masks.max() == FLARE22_LABEL_ENUM.BACK_GROUND.value

    def load_data_dict_into_dataset(self, data_dict, previous_data_dict):
        # Separate into emb and mask
        embedding = pick(data_dict, ["img_emb"])
        masks = omit(data_dict, ["img_emb", "original_size", "input_size", "n_masks"])

        # Check matched size
        _size = pick(data_dict, ["original_size", "input_size"])
        if self.input_size is None:
            self.input_size = _size["input_size"]
        if self.original_size is None:
            self.original_size = _size["original_size"]
            pass
        self._assert_size(_size)

        # Append into dataset
        previous_masks = omit(
            previous_data_dict, ["img_emb", "original_size", "input_size", "n_masks"]
        )

        for class_number, mask in masks.items():
            # Omit class that not in the list
            if int(class_number) not in self.class_selected:
                continue

            previous_mask = previous_masks.get(class_number, {}).get("mask", None)
            # If there are no previous mask -> abort
            if previous_mask is None:
                continue
            self.dataset.append(
                {
                    **embedding,
                    "mask": mask["mask"],
                    "previous_mask": previous_mask,
                    "class_number": class_number,
                }
            )
        pass

    def _assert_size(self, d):
        assert (
            self.input_size[0] == d["input_size"][0]
        ), f"Inconsistent input size: stored {self.input_size} but got {d['input_size']}"
        assert (
            self.input_size[1] == d["input_size"][1]
        ), f"Inconsistent Input size: stored {self.input_size} but got {d['input_size']}"

        assert (
            self.original_size[0] == d["original_size"][0]
        ), f"Inconsistent Original size: stored {self.original_size} but got {d['original_size']}"
        assert (
            self.original_size[1] == d["original_size"][1]
        ), f"Inconsistent Original size: stored {self.original_size} but got {d['original_size']}"
        pass

    def self_check(self):
        for idx, data in enumerate(self.dataset):
            assert data["img_emb"].shape == (
                1,
                256,
                64,
                64,
            ), f"{idx} - {data['img_emb'].shape}"
            pass
        pass


def load_model(checkpoint="./sam_vit_b_01ec64.pth", checkpoint_type="vit_b") -> Sam:
    sam: Sam = sam_model_registry[checkpoint_type](checkpoint=checkpoint)
    return sam


if __name__ == "__main__":
    sam = load_model()
    sam.to("cuda:0")
    FLARE22_MaskDrop.LIMIT = 20
    dataset = FLARE22_MaskDrop(
        cache_name=FLARE22_MaskDrop.VAL_CACHE_NAME,
        metadata_path=VAL_METADATA,
        is_debug=False,
        pre_trained_sam=sam,
        device=sam.device,
    )
    dataset.preprocess()
    dataset.preload()
    print(len(dataset))
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=True)
    for idx, batch in enumerate(loader):
        for k, v in batch.items():
            print(f"{k} : {v.shape}")
            if idx == 10:
                break
    pass
