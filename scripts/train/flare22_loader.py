import json

from tqdm import tqdm
import numpy as np
import os
import torch
from enum import Enum
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from scripts.train.sam_train import SamTrain
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry
from scripts.utils import (
    load_file_npz,
    load_img,
    make_directory,
    omit,
    pick,
)

# Respective to root
DATASET_ROOT = "./dataset/FLARE22-version1/"
TRAIN_PATH = "./dataset/FLARE22-version1/TrainImageProcessed"
TRAIN_MASK = "./dataset/FLARE22-version1/TrainMask"
TRAIN_METADATA = "./dataset/FLARE22-version1/train_metadata.json"


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


class SinglePointObjDetectFile:
    def __init__(
        self,
        train_metadata_path: str = TRAIN_METADATA,
        dataset_root: str = DATASET_ROOT,
    ) -> None:
        with open(train_metadata_path) as out:
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


class FLARE22(Dataset):
    # FOR DEBUG
    LIMIT = 10

    def __init__(
        self,
        pre_trained_sam: Sam = None,
        train_metadata_path: str = TRAIN_METADATA,
        dataset_root: str = DATASET_ROOT,
        cache_name: str = "single_point_object_detect",
        is_debug: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        # For preprocess data
        self.device = device
        self.pre_trained_sam = pre_trained_sam
        if pre_trained_sam:
            self.pre_trained_sam.eval()
            self.train_sam = SamTrain(self.pre_trained_sam)
        self.dataset_root = dataset_root

        self.file = SinglePointObjDetectFile(
            train_metadata_path=train_metadata_path, dataset_root=dataset_root
        )
        if is_debug:
            self.file.data = self.file.data[: self.LIMIT]
        self.cache_path = os.path.join(self.dataset_root, cache_name)
        make_directory(self.cache_path)
        self.dataset = []

        self.input_size = None
        self.original_size = None

    def preload(self, strict=False):
        for data in tqdm(
            self.file.data, desc="Preload data to RAM", total=len(self.file.data)
        ):
            cache_path = os.path.join(
                self.cache_path, f"{data['name']}", f"{data['id_number']}.pt"
            )
            if os.path.exists(cache_path):
                # Load the existed cached
                data_dict = torch.load(cache_path)
                self.load_data_dict_into_dataset(data_dict)
                continue

            if strict:
                raise FileNotFoundError(f"Can not found cache data: {cache_path}")
        pass

    def preprocess(self):
        """Efficient caching and broadcast cache into pair of embedding-mask"""
        if len(self.dataset) != 0:
            return
        # All data, sometime it has invalid mask for a target -> discard it
        for data in tqdm(
            self.file.data,
            desc="Preprocessing/Checking embedding",
            total=len(self.file.data),
        ):
            cache_path = os.path.join(
                self.cache_path, f"{data['name']}", f"{data['id_number']}.pt"
            )
            if os.path.exists(cache_path):
                continue
            _ = make_directory(cache_path, is_file=True)

            # If don't have the data dict for this img -> create it
            masks = load_file_npz(data["mask_path"])
            if self._is_valid_mask(masks):
                continue
            # Calculate the embedding
            img = load_img(data["img_path"])
            assert self.train_sam is not None, "Preprocess require the train-sam module"
            img_emb, original_size, input_size = self.train_sam.prepare_img(image=img)

            # Cache embedding and mask
            data_dict = {
                "img_emb": img_emb,
                "original_size": torch.as_tensor(original_size),
                "input_size": torch.as_tensor(input_size),
            }

            for _cls_value in np.unique(masks):
                data_dict[_cls_value] = torch.as_tensor(masks == _cls_value)
                pass
            data_dict["n_masks"] = np.unique(masks).shape[0]
            torch.save(data_dict, cache_path)
            # self.load_data_dict_into_dataset(data_dict)
        pass

    def _is_valid_mask(self, masks):
        """Filter out bad mask"""
        return masks.max() == FLARE22_LABEL_ENUM.BACK_GROUND.value

    def load_data_dict_into_dataset(self, data_dict):
        # Separate into emb and mask
        _emb = pick(data_dict, ["img_emb", "original_size", "input_size", "n_masks"])
        _masks = omit(data_dict, ["img_emb", "original_size", "input_size", "n_masks"])

        if self.input_size is None:
            self.input_size = _emb["input_size"]
        if self.original_size is None:
            self.original_size = _emb["original_size"]
        self._assert_equal(_emb)

        # Append into dataset
        for _, v in _masks.items():
            self.dataset.append({**_emb, "mask": v})
        pass

    def _assert_equal(self, d):
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

        return dict(
            img_emb=img_emb.to(self.device),
            mask=mask.to(self.device),
        )

    def __len__(self):
        return len(self.dataset)

    def get_size(self):
        input_size = (self.input_size[0], self.input_size[1])
        original_size = (self.original_size[0], self.original_size[1])
        return input_size, original_size


def load_model(checkpoint="./sam_vit_b_01ec64.pth", checkpoint_type="vit_b") -> Sam:
    sam: Sam = sam_model_registry[checkpoint_type](checkpoint=checkpoint)
    return sam


if __name__ == "__main__":
    sam = load_model()
    sam.to('cuda:0')
    FLARE22.LIMIT = 20
    dataset = FLARE22(is_debug=False, pre_trained_sam=sam)
    dataset.preprocess()
    print(len(dataset))
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=True)
    for batch in loader:
        for k, v in batch.items():
            print(f"{k} : {v.shape}")
            pass
        break

    pass
