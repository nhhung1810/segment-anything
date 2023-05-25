from typing import List

import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm
from scripts.datasets.constant import DEFAULT_DEVICE, FLARE22_LABEL_ENUM, IMAGE_TYPE
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.experiments.mask_aug.bidirection_inference import (
    InferenceEngine,
    parser,
    get_test_image,
    load_model,
)
from scripts.experiments.mask_aug.inference import merge_function
from scripts.sam_train import SamTrain
from scripts.tools.evaluation.loading import post_process
from scripts.utils import make_directory, torch_try_load


class MultiPointInferenceEngine(InferenceEngine):
    def __init__(
        self,
        sam_train: SamTrain,
        inference_save_dir: str,
        selected_class: List[int],
        device: str,
    ) -> None:
        super().__init__(sam_train, inference_save_dir, selected_class, device)

    def inference(self, image_file: str, gt_file: str):
        _, masks, cache_volume, [starts, ends] = self.load_volume(image_file, gt_file)
        # FIXME: this is just a skips
        # volumes = torch_try_load("volumes.pt", device="cpu", default_return=[])
        # proposal = self.propose_keyframe(starts, ends)
        volumes = []
        proposal = self.propose_keyframe(starts, ends)
        for start_point in tqdm(
            proposal, desc="Running ensemble...", total=len(proposal)
        ):
            predict_volume = self.inference_bidirection(
                masks=masks,
                cache_volume=cache_volume,
                start_point=start_point,
                starts=starts,
                ends=ends,
            )
            volumes.append(predict_volume)
            pass

        result = self.fusion(volumes)
        result = result.transpose(1, 2, 0).astype(np.uint8)
        post_process(pred=result, gt_file=gt_file, out_dir=self.inference_save_dir)
        pass

    def fusion(self, volumes: List[np.ndarray]) -> np.ndarray:
        return self.fusion_with_voting(volumes=volumes)

    def fusion_with_voting(self, volumes: List[np.ndarray]) -> np.ndarray:
        assert (
            len(volumes) % 2 != 0
        ), "Voting only available with odd-ensemble, for simple tie break"
        result = np.zeros(volumes[0].shape, dtype=np.uint8)
        buffer = np.stack(volumes, axis=0)
        for idx in range(volumes[0].shape[0]):
            layers: np.ndarray = buffer[:, idx, :, :]
            class_pred_dict = {}
            # Merge between layer of the same-class
            for class_num in self.selected_class:
                class_layer: np.ndarray = (layers == class_num).astype(np.uint8)
                # Odd-Voting, 2-3 vote counted as true, 1-0 vote count as false
                class_layer = (
                    np.sum(class_layer, axis=0, keepdims=False) > 1.0
                ).astype(np.uint8)

                class_pred_dict[class_num] = Tensor(class_layer, device="cpu")
                pass
            # Merge inter-class
            merged_frame: np.ndarray = merge_function(
                class_pred_list=class_pred_dict, device="cpu"
            ).numpy()
            result[idx] = merged_frame

        return result

    def fusion_with_union(self, volumes: List[np.ndarray]) -> np.ndarray:
        # Fusion multiple volume, frame by frame
        result = np.zeros(volumes[0].shape, dtype=np.uint8)
        # Layer by layer merging
        buffer = np.stack(volumes, axis=0)
        for idx in range(volumes[0].shape[0]):
            layers = buffer[:, idx, :, :]
            # result = merge_layer(layers)
            # option 1: Merge each class first then merge inter-class, for each frame.
            # -> num_class x num_start_point of comparison
            class_pred_dict = {}
            # Merge between layer of the same-class
            for class_num in self.selected_class:
                class_layer = layers == class_num
                # simple fusion by pooling
                class_layer = np.max(class_layer, axis=0, keepdims=False)
                class_pred_dict[class_num] = Tensor(class_layer, device="cpu")
                pass
            # Merge inter-class
            merged_frame: np.ndarray = merge_function(
                class_pred_list=class_pred_dict, device="cpu"
            ).numpy()
            result[idx] = merged_frame

        return result

    def propose_keyframe(self, starts: np.ndarray, ends: np.ndarray) -> List[int]:
        # Heuristic: zero and middle of liver?
        # middle of liver and middle of gall?
        return [
            # First point
            0,
            # Middle gall point
            int(
                (
                    starts[FLARE22_LABEL_ENUM.GALLBLADDER.value]
                    + ends[FLARE22_LABEL_ENUM.GALLBLADDER.value]
                )
                * 0.5
            ),
            # 2/3 liver point
            int(
                (
                    starts[FLARE22_LABEL_ENUM.LIVER.value]
                    + (
                        ends[FLARE22_LABEL_ENUM.LIVER.value]
                        - starts[FLARE22_LABEL_ENUM.LIVER.value]
                    )
                    * (2.0 / 3.0)
                )
            ),
        ]


if __name__ == "__main__":
    args = parser.parse_args()
    device = f"cuda:{args.cuda}" if "cuda" in DEFAULT_DEVICE else DEFAULT_DEVICE

    inference_save_dir = args.output_dir
    input_dir = args.input_dir
    label_dir = args.label_dir
    selected_class = args.selected_class or list(range(1, 14))
    use_cache = args.use_cache
    model_path = (
        args.checkpoint
        or "runs/mask-liver-first-augment/mask-drop-230519-012732/model-110.pt"
    )
    selected_class = [1, 9]
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
    images_path = [p for p in images_path if "0002" in p]
    labels_path = [p for p in labels_path if "0002" in p]
    model = load_model(model_path, device)
    sam_train = SamTrain(model)
    inference_engine = MultiPointInferenceEngine(
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
