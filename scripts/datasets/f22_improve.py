import enum
from glob import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from typing import Dict, List, Tuple

from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.datasets.constant import (
    BASE_PRETRAIN_PATH,
    DEFAULT_DEVICE,
    FIX_VALIDATION_SPLIT,
    IMAGE_TYPE,
    TRAIN_NON_PROCESSED,
)
from scripts.sam_train import SamTrain
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry


def find_organ_range(masks: np.ndarray, class_num: int) -> Tuple[int, int]:
    start_idx, end_idx = None, None
    r = range(masks.shape[0])
    # [0, len, 1)
    for forward_idx, reverse_idx in zip(r, r[::-1]):
        if start_idx == None:
            start_idx = (
                forward_idx if (masks[forward_idx, ...] == class_num).any() else None
            )

        if end_idx == None:
            end_idx = (
                reverse_idx if (masks[reverse_idx, ...] == class_num).any() else None
            )

        if start_idx != None and end_idx != None:
            return start_idx, end_idx

    return start_idx, end_idx


def get_all_organ_range(masks) -> Tuple[np.ndarray, np.ndarray]:
    starts, ends = [], []
    for class_num in range(1, 14, 1):
        start_idx, end_idx = find_organ_range(masks=masks, class_num=class_num)
        starts.append(start_idx)
        ends.append(end_idx)

    # Padding zero for background pixel
    return np.array([0, *starts]), np.array([0, *ends])


def assert_paths(paths: List[Tuple[str, str]]):
    for (
        a,
        b,
    ) in paths:
        assert os.path.exists(a), a
        assert os.path.exists(b), b


def assert_equal(src, dest, skip_none=True):
    if src is None and skip_none:
        return dest
    assert src == dest, f"Need to be equal: {src} == {dest}"
    return dest


def get_patient_name(path):
    return os.path.basename(path).replace(".nii.gz", "")


def get_patient_num(image_path):
    return os.path.basename(image_path).replace("_0000.nii.gz", "")[-4:]


def get_train_test_split(group):
    assert group in ["train", "validation"]
    image_paths = glob(f"{TRAIN_NON_PROCESSED}/images/*.nii.gz")
    if group == "train":
        image_paths = [
            p for p in image_paths if get_patient_num(p) not in FIX_VALIDATION_SPLIT
        ]
        pass
    else:
        image_paths = [
            p for p in image_paths if get_patient_num(p) in FIX_VALIDATION_SPLIT
        ]
        pass
    label_paths = [
        os.path.join(
            f"{TRAIN_NON_PROCESSED}/labels",
            os.path.basename(p).replace("_0000.nii.gz", ".nii.gz"),
        )
        for p in image_paths
    ]
    return list(zip(image_paths, label_paths))


class F22_MaskPropagate(Dataset):
    LIMIT = 20
    TRAIN_CACHE_NAME = "f22-mp-improve/train"
    VAL_CACHE_NAME = "f22-mp-improve/validation"

    def __init__(
        self,
        paths: List[Tuple[str, str]] = [],
        pretrain_model: Sam = None,
        # cache_path: str = TRAIN_CACHE_NAME,
        is_training: bool = True,
        selected_class: List[int] = list(range(1, 14)),
        direction: Tuple[int, int] = [1, 1],
        n_frame: int = 2,
    ) -> None:
        self.volume_loader = FLARE22_Preprocess()
        self.direction = direction
        self.n_frame = n_frame
        self.selected_class = (selected_class,)

        if not is_training:
            self.cache_path = self.VAL_CACHE_NAME
            self.paths = get_train_test_split("validation")
            pass
        else:
            self.cache_path = self.TRAIN_CACHE_NAME
            self.paths = get_train_test_split("train")

        if pretrain_model:
            self.pretrain_model = pretrain_model
            self.pretrain_model.eval()
            self.train_sam = SamTrain(self.pretrain_model)

        self.input_size = None
        self.output_size = None
        self.dataset = []
        self.cache_size_path = os.path.join(self.cache_path, "size-data.pt")
        assert_paths(paths)

        pass

    def __getitem__(self, index):
        data = self.dataset[index]
        data = self.augment(data)
        data = self.format_batch(data)
        return data

    def augment(self, data):
        return data

    def format_batch(self, data: Dict[str, torch.Tensor]):
        # Remove the batch dimension
        img_emb = data["img_emb"][0]
        mask = data["mask"]
        previous_mask = (data["previous_mask"],)
        class_number = data["class_number"]
        return dict(
            img_emb=img_emb.to(self.device),
            mask=mask.to(self.device),
            previous_mask=previous_mask.to(self.device),
            class_number=class_number.to(self.device),
        )

    def __len__(self):
        return len(self.dataset)

    def preprocess(self):
        for image_path, label_path in tqdm(
            self.paths, desc="Checking/Preprocessing volume..."
        ):
            cache_path = os.path.join(
                self.cache_path, f"{get_patient_name(label_path)}.pt"
            )
            # Skip computation...
            if os.path.exists(cache_path):
                continue
            assert (
                self.train_sam is not None
            ), f"Need pre-train ViT model to compute the embedding"

            images, masks = self.volume_loader.run_with_config(
                image_file=image_path,
                gt_file=label_path,
                config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
            )
            all_emb = []
            for idx in trange(
                images.shape[0],
                desc="Calculate image embedding...",
                total=images.shape[0],
                leave=False,
            ):
                img = images[idx][..., None].repeat(3, -1)
                # img_emb.shape = [1, 256, 64, 64]
                img_emb, original_size, input_size = self.train_sam.prepare_img(
                    image=img
                )
                # Ensure the size data to be consistent
                self.original_size = assert_equal(self.original_size, original_size)
                self.input_size = assert_equal(self.original_size, input_size)
                all_emb.append(img_emb)
                pass
            all_emb = torch.cat(all_emb, dim=0)

            cache_data = {
                # "images": images,
                "img_emb": torch.as_tensor(all_emb),
                "masks": torch.as_tensor(masks),
            }
            torch.save(cache_data, cache_path)

            pass

        # Save the caching
        if self.original_size is not None or self.input_size is not None:
            torch.save(
                dict(original_size=original_size, input_size=input_size),
                self.cache_size_path,
            )
        else:
            # If all computing skipped -> load from cache...
            cache_data = torch.load(self.cache_size_path)
            self.original_size = cache_data["original_size"]
            self.input_size = cache_data["input_size"]
            pass

    def preload(self):
        # Need to packing up data with the previous frame
        for _, label_path in tqdm(self.paths, desc="Preprocessing volume..."):
            cache_path = os.path.join(
                self.cache_path, f"{get_patient_name(label_path)}.pt"
            )
            cache_data = torch.load(cache_path)
            embeddings = cache_data["img_emb"]
            masks = cache_data["masks"]
            starts, ends = get_all_organ_range(masks)
            # Iterate over each class
            for class_num in range(self.selected_class):
                start, end = starts[class_num], ends[class_num]
                # Trim down to head and tail
                organ_embed = embeddings[start : end + 1]
                organ_mask = masks[start : end + 1]
                # Extract the label and input, perform pairing
                organ_buffer = self.extract_organ_data(
                    organ_embed, organ_mask, class_num
                )
                # Load all to RAM
                self.dataset.extend(organ_buffer)
                pass
        pass

    def extract_organ_data(
        self, embeddings: torch.Tensor, masks: torch.Tensor, class_num: int
    ):
        # the embedding and masks are trimmed
        # to the head-tail of the class
        buffer_length = embeddings.shape[0]
        current_buffer = []
        for idx in trange(
            buffer_length,
            desc="Create current-buffer...",
            total=buffer_length,
            leave=False,
        ):
            current_emb = embeddings[idx]
            current_mask = masks[idx] == class_num
            current_buffer.append((current_emb, current_mask))
            pass

        # Inclusive step
        n_step_backward = self.direction[0] * self.n_frame
        n_step_forward = self.direction[1] * self.n_frame
        # Window-ing pairing
        result_buffer = []
        for idx in trange(
            buffer_length, desc="Pairing...", total=buffer_length, leave=False
        ):
            for previous_frame_idx in range(
                start=max(0, idx - n_step_backward),
                stop=min(buffer_length, idx + n_step_forward + 1),
            ):
                if previous_frame_idx == idx:
                    continue

                result_buffer.append(
                    {
                        "img_emb": current_buffer[idx][0],
                        "mask": current_buffer[idx][1],
                        "previous_mask": current_buffer[previous_frame_idx][1],
                        "class_number": torch.LongTensor([class_num]),
                    }
                )
                pass
            pass
        return result_buffer


if __name__ == "__main__":
    sam: Sam = sam_model_registry["vit_b"](checkpoint=BASE_PRETRAIN_PATH)
    sam.to(DEFAULT_DEVICE)
    dataset = F22_MaskPropagate(
        is_training=False,
        pretrain_model=sam,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    for idx, batch in enum(loader):
        if idx > 5:
            break
        print(batch)
        pass
    pass
