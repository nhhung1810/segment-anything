from argparse import ArgumentParser

from glob import glob
import os
from typing import Dict, List

import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
from scripts.datasets.constant import (
    DEFAULT_DEVICE,
    TEST_NON_PROCESSED,
    IMAGE_TYPE,
    FLARE22_LABEL_ENUM,
)
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.experiments.mask_aug.inference import (
    get_all_organ_range,
    merge_function,
    load_model,
    pick_best_mask,
)
from scripts.sam_train import SamTrain
from scripts.tools.evaluation.loading import post_process
from scripts.utils import make_directory, torch_try_load

CACHE_VOLUME_TYPE = Dict[int, Dict[str, Tensor]]


class InferenceEngine:
    def __init__(
        self,
        sam_train: SamTrain,
        inference_save_dir: str,
        selected_class: List[int],
        device: str,
    ) -> None:
        self.sam_train = sam_train
        self.device: str = device
        self.inference_save_dir = inference_save_dir
        self.selected_class = selected_class
        self.flare22_preprocessor = FLARE22_Preprocess()
        pass

    def load_volume(self, image_file: str, gt_file: str):
        volume, masks = self.flare22_preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )
        starts, ends = get_all_organ_range(masks)
        cache_volume_path = image_file.replace(".nii.gz", ".cache.pt")
        cache_volume = torch_try_load(cache_volume_path, device=self.device)
        if cache_volume is None:
            cache_volume = self.get_image_embedding(volume)
            torch.save(cache_volume, cache_volume_path)

        return volume, masks.copy(), cache_volume, [starts, ends]

    def inference_class(self, cache_frame: Dict[str, Tensor], previous_mask: Tensor):
        img_emb = cache_frame["img_emb"]
        original_size = cache_frame["original_size"]
        input_size = cache_frame["input_size"]
        mask_input = torch.as_tensor(previous_mask)

        _, _, _, mask_input_torch = self.sam_train.prepare_prompt(
            original_size=original_size, mask_input=mask_input[None, ...]
        )

        mask_pred, _, _ = self.sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Get the logits to compute the stability
            return_logits=False,
            mask_input=mask_input_torch,
        )
        chosen_idx, _ = pick_best_mask(
            pred_multi_mask=mask_pred,
            previous_mask=previous_mask,
            gt_binary_mask=None,
            device=self.device,
            strategy="prev",
        )
        mask_pred = mask_pred[0, chosen_idx]
        return mask_pred

    def inference_frame(
        self,
        cache_frame: Dict[str, Tensor],
        previous_mask: Tensor,
        starts: np.ndarray,
        ends: np.ndarray,
        frame_idx: int,
        masks: np.ndarray,
        is_init_dict: Dict[int, bool],
    ) -> np.ndarray:
        class_pred_dict = {}
        for class_num in self.selected_class:
            if frame_idx not in range(starts[class_num], ends[class_num] + 1):
                # Skip class that not in the range
                class_pred_dict[class_num] = torch.zeros(
                    cache_frame["original_size"], device=self.device
                )
                continue
            # If this mask is the first time inference for this class
            if not is_init_dict.get(class_num, False):
                class_mask = masks[frame_idx] == class_num
                is_init_dict[class_num] = True
            else:
                class_mask = previous_mask == class_num
            class_pred = self.inference_class(
                cache_frame=cache_frame,
                previous_mask=class_mask,
            )
            class_pred_dict[class_num] = class_pred
            pass

        merged_pred = merge_function(class_pred_dict)
        return merged_pred.detach().cpu().numpy().astype(np.uint8)

    def get_image_embedding(self, volume) -> CACHE_VOLUME_TYPE:
        volume_embedding = {}
        for idx in range(volume.shape[0]):
            img_emb, original_size, input_size = self.sam_train.prepare_img(
                image=volume[idx]
            )
            cache_frame: Dict[str, Tensor] = {
                "img_emb": img_emb,
                "original_size": original_size,
                "input_size": input_size,
            }

            volume_embedding[idx] = cache_frame
            pass

        return volume_embedding

    def inference_bidirection(
        self,
        # volume: np.ndarray,
        masks: np.ndarray,
        cache_volume: Dict[int, Dict[str, Tensor]],
        start_point: int,
        starts: np.ndarray,
        ends: np.ndarray,
    ):
        save_pred = []
        t_forward = range(start_point, len(cache_volume), 1)
        t_backward = range(start_point - 1, -1, -1)
        is_init_dict = {}
        # go forward
        for idx in tqdm(
            t_forward, desc="Forward-ing...", total=len(t_forward), leave=False
        ):
            if idx == start_point:
                previous_mask = masks[idx].astype(np.uint8)
            pred = self.inference_frame(
                cache_frame=cache_volume[idx],
                previous_mask=previous_mask,
                starts=starts,
                ends=ends,
                frame_idx=idx,
                masks=masks,
                is_init_dict=is_init_dict,
            )
            previous_mask = pred
            save_pred.append(pred)
            pass

        # go backward
        for idx in tqdm(
            t_backward, desc="Backward-ing...", total=len(t_backward), leave=False
        ):
            # Take the first mask is correct understand as 'previous' pred
            previous_mask = save_pred[0]
            pred = self.inference_frame(
                cache_frame=cache_volume[idx],
                previous_mask=previous_mask,
                starts=starts,
                ends=ends,
                frame_idx=idx,
                masks=masks,
                is_init_dict=is_init_dict,
            )
            save_pred.insert(0, pred)
            pass

        return np.stack(save_pred, axis=0)

    def propose_keyframe(self, starts: np.ndarray, ends: np.ndarray):
        # Heuristic: Liver keyframe 1/2
        # return 0
        return int(
            (
                starts[FLARE22_LABEL_ENUM.LIVER.value]
                + ends[FLARE22_LABEL_ENUM.LIVER.value]
            )
            / 2.0
        )

    def inference(self, image_file: str, gt_file: str):
        # load data, decide the start point for propagation
        _, masks, cache_volume, [starts, ends] = self.load_volume(image_file, gt_file)
        start_point = self.propose_keyframe(starts, ends)
        # propagation bi-direction
        predict_volume = self.inference_bidirection(
            masks=masks,
            cache_volume=cache_volume,
            start_point=start_point,
            starts=starts,
            ends=ends,
        )

        # save inference
        predict_volume = predict_volume.transpose(1, 2, 0).astype(np.uint8)
        post_process(
            pred=predict_volume,
            gt_file=gt_file,
            out_dir=self.inference_save_dir,
        )
        pass

    pass


parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="Volume image directory",
    default=f"{TEST_NON_PROCESSED}/images",
)
parser.add_argument(
    "--cuda",
    type=int,
    help="GPU index",
    default=1,
)
parser.add_argument(
    "-l",
    "--label_dir",
    type=str,
    help="Volume label directory",
    default=f"{TEST_NON_PROCESSED}/labels",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Output directory",
    default="runs/submission/pred",
)
parser.add_argument(
    "-ckpt", "--checkpoint", type=str, help="Model checkpoint to load", default=None
)
parser.add_argument(
    "--use_cache",
    type=str,
    help="Allow the model to use embedding cache for faster inference",
    default=True,
)
parser.add_argument(
    "--selected_class",
    nargs="+",
    type=int,
    help="List of class, no pre/pos-fix separated by space, i.e. 1 2 3",
    default=None,  # for liver and gallbladder
)


def get_test_image(input_dir, label_dir):
    images_path: List[str] = sorted(glob(f"{input_dir}/*.nii.gz"))
    images_path = [os.path.basename(p) for p in images_path]
    # images_path = [os.path.basename(p) for p in images_path if "0002" in p]
    # By some way, they don't have gallbladder, which i will omit for now
    images_path.remove("FLARETs_0006_0000.nii.gz")
    images_path.remove("FLARETs_0008_0000.nii.gz")
    images_path.remove("FLARETs_0021_0000.nii.gz")
    images_path.remove("FLARETs_0031_0000.nii.gz")
    images_path.remove("FLARETs_0033_0000.nii.gz")
    images_path.remove("FLARETs_0036_0000.nii.gz")
    images_path.remove("FLARETs_0038_0000.nii.gz")
    images_path.remove("FLARETs_0043_0000.nii.gz")
    images_path.remove("FLARETs_0044_0000.nii.gz")
    images_path.remove("FLARETs_0048_0000.nii.gz")
    labels_path = [
        os.path.join(label_dir, p.replace("_0000.nii.gz", ".nii.gz"))
        for p in images_path
    ]
    images_path = [os.path.join(input_dir, p) for p in images_path]

    for i, p in zip(images_path, labels_path):
        assert os.path.exists(i), f"{i}"
        assert os.path.exists(p), f"{p}"
    return images_path, labels_path


if __name__ == "__main__":
    args = parser.parse_args()
    device = f"cuda:{args.cuda}" if "cuda" in DEFAULT_DEVICE else DEFAULT_DEVICE

    inference_save_dir = args.output_dir
    input_dir = args.input_dir
    label_dir = args.label_dir
    selected_class = args.selected_class or list(range(1, 14))
    use_cache = args.use_cache
    model_path = args.checkpoint or "runs/mask-liver-first-augment/mask-drop-230519-012732/model-170.pt"

    for c in selected_class:
        assert c in range(1, 14)
    class_name = [
        value.name.lower().replace("_", " ")
        for value in FLARE22_LABEL_ENUM
        if value.value in selected_class
    ]
    print(
        f"""
        model-path     : {model_path}
        save-dir       : {inference_save_dir}
        input-dir      : {input_dir}
        label-dir      : {label_dir}
        is-use-cache   : {use_cache}
        selected-class : {class_name}
        """
    )
    make_directory(inference_save_dir)
    images_path, labels_path = get_test_image(input_dir, label_dir)
    model = load_model(model_path, device)
    sam_train = SamTrain(model)
    inference_engine = InferenceEngine(
        sam_train=sam_train,
        inference_save_dir=inference_save_dir,
        selected_class=selected_class,
        device=device,
    )
    for image_file, label_file in tqdm(
        zip(images_path, labels_path),
        total=len(images_path),
        desc="Per-patient inference...",
    ):
        inference_engine.inference(image_file=image_file, gt_file=label_file)
    pass
