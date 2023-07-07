import enum
from glob import glob
import os
from time import time_ns
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from typing import Callable, Dict, List, Tuple

from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.datasets.constant import (
    BASE_PRETRAIN_PATH,
    DATASET_ROOT,
    DEFAULT_DEVICE,
    FIX_VALIDATION_SPLIT,
    IMAGE_TYPE,
    TRAIN_NON_PROCESSED,
)
from scripts.sam_train import SamTrain
from scripts.utils import make_directory, omit
from segment_anything.modeling.sam import Sam
from segment_anything.build_sam import sam_model_registry
import albumentations as A


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


def create_gaussian_2d_kernel(length: int, sigma: float, device: str):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = torch.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length, device=device)
    gauss = torch.exp(
        -0.5 * torch.square(ax) / torch.square(torch.as_tensor(sigma, device=device))
    )
    kernel = torch.outer(gauss, gauss)
    return kernel / torch.sum(kernel)


def gaussian_filter(input: torch.Tensor, kernel_length: int, sigma: float):
    kernel = create_gaussian_2d_kernel(kernel_length, sigma, input.device)
    result = torch.nn.functional.conv2d(input, kernel, padding="same")
    return result


class Augmentation:
    def __init__(self, aug_config: Dict[int, object], seed=None) -> None:
        self.selected_class = []
        self.aug_fn_map = {}
        self.seed = seed or (time_ns() % (2**32 - 1))
        self.random_state = np.random.RandomState(seed=self.seed)
        if aug_config is not None:
            self.selected_class: List[int] = list(aug_config.keys())
            # Prebuilt augmentation mapping
            self.aug_fn_map = {
                class_num: self.build_augmentation(config=aug_config[class_num])
                for class_num in self.selected_class
            }

        pass

    def apply(self, previous_mask, class_number):
        # This function should not throw error and cancel the training
        try:
            if class_number not in self.selected_class:
                return previous_mask

            return self.aug_fn_map[class_number](previous_mask)
        except Exception as msg:
            print(msg)
            return previous_mask

    def build_augmentation(
        self, config: dict
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        key = config["key"]
        aug_params = omit(config, ["key"])
        if key == "one-block-drop":
            fn = lambda previous_mask: self.one_block_drop(previous_mask, **aug_params)
            return fn
        elif key == "pixel-drop":
            fn = lambda previous_mask: self.pixel_drop(previous_mask, **aug_params)
            return fn
        elif key == "pick-square":
            fn = lambda previous_mask: self.pick_square(previous_mask, **aug_params)

        return lambda x: x

    def calculate_bbox(self, coors: torch.Tensor):
        xs = coors[:, 1]
        ys = coors[:, 0]
        top_left = [xs.min(), ys.min()]
        right_bottom = [xs.max(), ys.max()]
        w, h = [right_bottom[0] - top_left[0], right_bottom[1] - top_left[1]]
        return top_left, right_bottom, w, h

    def pick_square(
        self,
        mask: torch.Tensor,
        center_radius: int = 3,
        radius_width: int = 5,
        gaussian_config=None,
        augmentation_prop: float = 0.5,
    ):
        # binary mask input
        if self.random_state.uniform(0, 1) > augmentation_prop:
            return mask
        # Pick the pixel
        coors = torch.argwhere(mask)
        idx = self.random_state.randint(low=0, high=coors.shape[0])
        x = coors[idx][0].item()
        y = coors[idx][1].item()
        # Generate the square by radius
        result = torch.zeros(mask.shape)
        radius = torch.clamp(
            torch.as_tensor(int(center_radius + self.random_state.uniform(-1, 1) * radius_width)),
            min=torch.as_tensor(1),
            max=torch.as_tensor(center_radius + radius_width),
        )
        xmax = min(x + radius, mask.shape[0])
        ymax = min(y + radius, mask.shape[1])
        xmin = max(x - radius, 0)
        ymin = max(y - radius, 0)
        result[xmin:xmax, ymin:ymax] = 1.0

        if not gaussian_config:
            return result
        if self.random_state.uniform(0, 1) > gaussian_config["prob"]:
            return result

        result = gaussian_filter(
            result,
            sigma=gaussian_config["sigma"],
            kernel_length=gaussian_config["kernel_length"],
        )
        result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
        return result

    def one_block_drop(
        self,
        previous_mask: torch.Tensor,
        max_crop_ratio: float,
        augmentation_prop: float,
    ):
        coors = torch.argwhere(previous_mask)
        if coors.shape[0] == 0:
            return previous_mask
        top_left, right_bottom, w, h = self.calculate_bbox(coors)
        max_crop = int(min(w, h) * max_crop_ratio)
        if max_crop == 0:
            return previous_mask

        drop_fn = A.CoarseDropout(
            max_holes=1,
            max_height=int(max_crop),
            max_width=int(max_crop),
            fill_value=0.0,
            p=augmentation_prop,
        )
        # Sub-region augmentation only
        previous_mask = previous_mask.clone()
        y_slice = slice(top_left[1], right_bottom[1])
        x_slice = slice(top_left[0], right_bottom[0])
        sub_mask = previous_mask[y_slice, x_slice]
        sub_mask = drop_fn(image=sub_mask.cpu().numpy())["image"]
        previous_mask[y_slice, x_slice] = torch.as_tensor(sub_mask)

        return previous_mask

    def pixel_drop(
        self,
        previous_mask: torch.Tensor,
        drop_out_prop: float,
        augmentation_prop: float,
    ):
        coors = torch.argwhere(previous_mask)
        if coors.shape[0] == 0:
            return previous_mask
        top_left, right_bottom, w, h = self.calculate_bbox(coors)
        # Just a safety number, as I think that 4 pixels
        # is not enough for augmentation
        if min(w, h) < 3:
            return previous_mask
        drop_fn = A.PixelDropout(
            dropout_prob=drop_out_prop, drop_value=0, p=augmentation_prop
        )
        # Sub-region augmentation only
        previous_mask = previous_mask.clone()
        y_slice = slice(top_left[1], right_bottom[1])
        x_slice = slice(top_left[0], right_bottom[0])
        sub_mask = previous_mask[y_slice, x_slice]
        sub_mask = drop_fn(image=sub_mask.cpu().numpy())["image"]
        previous_mask[y_slice, x_slice] = torch.as_tensor(sub_mask)
        return previous_mask


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
        device: str = DEFAULT_DEVICE,
        aug_config: Dict[int, object] = None,
    ) -> None:
        self.volume_loader = FLARE22_Preprocess()
        self.direction = direction
        self.n_frame = n_frame
        self.selected_class = selected_class
        self.device = device
        self.is_training = is_training

        if not is_training:
            self.cache_path = os.path.join(DATASET_ROOT, self.VAL_CACHE_NAME)
            self.paths = get_train_test_split("validation")
            pass
        else:
            self.cache_path = os.path.join(DATASET_ROOT, self.TRAIN_CACHE_NAME)
            self.paths = get_train_test_split("train")

        if pretrain_model:
            self.pretrain_model = pretrain_model
            self.pretrain_model.eval()
            self.train_sam = SamTrain(self.pretrain_model)

        self.original_size = None
        self.input_size = None
        self.dataset = []
        self.cache_size_path = os.path.join(self.cache_path, "size-data.pt")
        make_directory(self.cache_size_path, is_file=True)
        assert_paths(paths)

        # Augmentation engine
        self.augmentation_engine = Augmentation(aug_config=aug_config)

        pass

    def __getitem__(self, index):
        data = self.dataset[index]
        # Buffer data should not be modified
        result = self.augment(data)
        result = self.format_batch(result)
        return result

    def augment(self, data):
        class_number: int = data["int_class_number"]
        previous_mask: torch.Tensor = data["previous_mask"]
        previous_mask = self.augmentation_engine.apply(
            previous_mask=previous_mask, class_number=class_number
        )
        result = omit(data, ["previous_mask"])
        result["previous_mask"] = previous_mask
        return result

    def format_batch(self, data: Dict[str, torch.Tensor]):
        img_emb = data["img_emb"]
        mask = data["mask"]
        previous_mask = data["previous_mask"]
        class_number = data["class_number"]

        return dict(
            img_emb=img_emb.to(self.device),
            mask=mask.to(self.device),
            previous_mask=previous_mask.to(self.device),
            class_number=class_number.to(self.device),
        )

    def __len__(self):
        return len(self.dataset)

    def get_size(self):
        return (self.input_size[0], self.input_size[1]), (
            self.original_size[0],
            self.original_size[1],
        )

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
                self.input_size = assert_equal(self.input_size, input_size)
                all_emb.append(img_emb)
                pass
            all_emb = torch.cat(all_emb, dim=0)

            cache_data = {
                # "images": images,
                "img_emb": torch.as_tensor(all_emb),
                "masks": torch.as_tensor(masks.astype(np.uint8)),
            }
            torch.save(cache_data, cache_path)

            pass

        # Save the caching
        if self.original_size is not None and self.input_size is not None:
            torch.save(
                dict(original_size=original_size, input_size=input_size),
                self.cache_size_path,
            )

    def preload(self):
        if self.original_size is None or self.input_size is None:
            # If all computing skipped -> load from cache...
            cache_data = torch.load(self.cache_size_path)
            self.original_size = cache_data["original_size"]
            self.input_size = cache_data["input_size"]
            pass
        # Need to packing up data with the previous frame
        for _, label_path in tqdm(self.paths, desc="Preloading, per-patient..."):
            cache_path = os.path.join(
                self.cache_path, f"{get_patient_name(label_path)}.pt"
            )
            cache_data = torch.load(cache_path)
            embeddings = cache_data["img_emb"]
            masks = cache_data["masks"]
            starts, ends = get_all_organ_range(masks)
            # Iterate over each class
            for class_num in self.selected_class:
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
                max(0, idx - n_step_backward),
                min(buffer_length, idx + n_step_forward + 1),
            ):
                if previous_frame_idx == idx:
                    continue

                result_buffer.append(
                    {
                        "img_emb": current_buffer[idx][0],
                        "mask": current_buffer[idx][1],
                        "previous_mask": current_buffer[previous_frame_idx][1],
                        "class_number": torch.LongTensor([class_num]),
                        "int_class_number": class_num,
                    }
                )
                pass
            pass
        return result_buffer


if __name__ == "__main__":
    sam: Sam = sam_model_registry["vit_b"](checkpoint=BASE_PRETRAIN_PATH)
    sam.to(DEFAULT_DEVICE)
    dataset = F22_MaskPropagate(
        is_training=True,
        pretrain_model=sam,
        direction=[1, 1],
        n_frame=3,
    )
    dataset.preprocess()
    dataset.preload()
    print(f"Dataset loaded, in training? {dataset.is_training}, size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    for idx, batch in enumerate(loader):
        if idx > 5:
            break
        for k, v in batch.items():
            print(k, v.shape)
        pass
    pass
