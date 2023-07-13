import numpy as np
import torch
from torch import Tensor
from scripts.experiments.beam_search.components import BeamSearchOptionData, Tracing

from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam

import os
from scipy.ndimage import gaussian_filter
from scripts.datasets.constant import IMAGE_TYPE
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.experiments.mask_aug.inference import get_all_organ_range, pick_best_mask
from scripts.sam_train import SamTrain
from scripts.utils import torch_try_load
from time import time_ns
from typing import Callable, Dict, List

preprocessor = FLARE22_Preprocess()
FILTER_TYPE = Callable[[BeamSearchOptionData], bool]


def pick_one_pixel(mask: torch.Tensor, radius=3, seed=10, gaussian_config=None):
    # binary mask input
    coors = torch.argwhere(mask)
    idx = np.random.RandomState(seed=seed).randint(low=0, high=coors.shape[0])
    x = coors[idx][0].item()
    y = coors[idx][1].item()
    result = np.zeros(mask.shape)
    xmax = min(x + radius, mask.shape[0])
    ymax = min(y + radius, mask.shape[1])
    xmin = max(x - radius, 0)
    ymin = max(y - radius, 0)
    result[xmin:xmax, ymin:ymax] = 1.0

    if gaussian_config:
        result = gaussian_filter(result, sigma=gaussian_config["sigma"])
        result = (result - np.min(result)) / (np.max(result) - np.min(result))

    result = torch.as_tensor(result)
    return result


def make_gauss_point_mask(x, y, shape=(512, 512), radius=3, gaussian_config=None):
    result = np.zeros(shape)
    xmax = min(x + radius, shape[1])
    ymax = min(y + radius, shape[0])
    xmin = max(x - radius, 0)
    ymin = max(y - radius, 0)
    result[ymin:ymax, xmin:xmax] = 1.0

    if gaussian_config:
        result = gaussian_filter(result, sigma=gaussian_config["sigma"])
        result = (result - np.min(result)) / (np.max(result) - np.min(result))

    result = torch.as_tensor(result)
    return result


def make_centroid(
    input_masks, local_mean_centroid: np.ndarray, is_input_reversed=False
):
    # `is_input_reversed: bool`: it's proven that reversed input have more consistent feature.
    # Therefore, our feature is compute on reversed input. If the input is not reversed,
    # set `is_input_reversed=True` so it will be done for you

    if not is_input_reversed:
        masks = input_masks[::-1]
    else:
        masks = input_masks

    starts, ends = get_all_organ_range(masks)
    starts = np.nan_to_num(starts.astype(np.float32), nan=0.0)
    ends = np.nan_to_num(ends.astype(np.float32), nan=0.0)
    dur = ends - starts
    dur: np.ndarray = np.pad(
        np.reshape(dur, [-1, 1]), pad_width=[(0, 0), (0, 2)], constant_values=512
    )
    # Take the centroid in z-axis of local organ, and then add back the starts
    proposal = dur * local_mean_centroid + np.pad(
        np.reshape(starts, [-1, 1]), pad_width=[(0, 0), (0, 2)], constant_values=0.0
    )

    if not is_input_reversed:
        # reverse the z-axis via subtraction
        proposal[:, 0] = masks.shape[0] - proposal[:, 0]
        pass

    return np.ceil(proposal).astype(np.uint16)


class BeamSearchInferenceEngine:
    LOCAL_MEAN_CENTROID = np.zeros((14, 3))
    LOCAL_MEAN_CENTROID[1] = np.array([0.38626369, 0.54490201, 0.67218827])

    def __init__(
        self,
        volumes: Tensor,
        caches: Tensor,
        masks: Tensor,
        sam_train: SamTrain,
        stability_config: Dict[str, object],
        gaussian_config: Dict[str, object],
        start_radius: float,
        seed: int = None,
        strategy_name: str = "",
        allow_evolution: bool = False,
    ) -> None:
        self.volumes = volumes
        self.caches = caches
        self.masks = masks
        self.seed = seed or time_ns() % (2**32 - 1)
        self.gaussian_config = gaussian_config
        self.start_radius = int(start_radius)
        self.strategy_name = strategy_name
        self.allow_evolution = allow_evolution
        self.sam_train = sam_train
        self.stability_config = self.prepare_default_stability_config(stability_config)
        self.starts, self.ends = get_all_organ_range(masks)
        pass

    @staticmethod
    def make_default_config() -> Dict[str, object]:
        return {
            "stability_config": None,
            "start_radius": 3,
            "seed": None,
            "gaussian_config": {"sigma": 10.0},
            "strategy_name": "local-mean-centroid",
            "allow_evolution": True,
        }

    def prepare_default_stability_config(self, stability_config: Dict[str, object]):
        stability_config = stability_config or {}
        stability_config["threshold_start"] = stability_config.get(
            "threshold_start", 0.1
        )
        stability_config["threshold_end"] = stability_config.get("threshold_end", 0.9)
        stability_config["threshold_num"] = stability_config.get("threshold_num", 10)
        stability_config["offset"] = stability_config.get("offset", 0.1)
        return stability_config

    @torch.no_grad()
    def core_inference(self, idx: int, previous_mask: Tensor):
        original_size = self.caches[idx]["original_size"]
        input_size = self.caches[idx]["input_size"]
        img_emb = self.caches[idx]["img_emb"]

        _, _, _, mask_input_torch = self.sam_train.prepare_prompt(
            original_size=original_size, mask_input=previous_mask[None, ...]
        )

        mask_logits, _, _ = self.sam_train.predict_torch(
            image_emb=img_emb,  # 1, 256, 64, 64
            input_size=input_size,
            original_size=original_size,
            multimask_output=True,
            # Get the logits to compute the stability
            return_logits=True,
            mask_input=mask_input_torch,
        )
        mask_logits = mask_logits.cpu()

        mask_pred = mask_logits > 0.0

        chosen_idx, _ = pick_best_mask(
            pred_multi_mask=mask_pred,
            previous_mask=previous_mask,
            gt_binary_mask=None,
            device="cpu",
            strategy="prev",
        )

        return chosen_idx, mask_logits

    def confidence_score(self, mask_logits, threshold):
        # Idea: confidence is high when the prob of
        # foreground high and prob of background is low
        sigmoid_mask = torch.sigmoid(mask_logits).numpy()
        foreground_score = self.safe_mean(sigmoid_mask[sigmoid_mask >= threshold])
        background_score = 1.0 - self.safe_mean(sigmoid_mask[sigmoid_mask < threshold])
        return np.mean([foreground_score, background_score])

    def safe_mean(self, x) -> np.ndarray:
        return np.nan_to_num(np.mean(x), 0.0)

    def calculate_stability_score_with_sigmoid(
        self, masks: torch.Tensor, mask_threshold: float, threshold_offset: float
    ) -> torch.Tensor:
        """
        Exactly like the `calculate_stability_score`, but using sigmoid for better scale
        """
        sigmoid_masks = torch.sigmoid(masks)
        intersections = (
            (sigmoid_masks > (mask_threshold + threshold_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        unions = (
            (sigmoid_masks > (mask_threshold - threshold_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        return intersections / unions

    def generate_option(
        self, option: BeamSearchOptionData, is_forward=True
    ) -> List[BeamSearchOptionData]:
        previous_mask = option.get_mask()
        chosen_idx, mask_logits = self.core_inference(
            idx=option.next_frame_idx, previous_mask=previous_mask
        )

        options = self.generate_next(
            current_chosen_mask_logits=mask_logits[0, chosen_idx],
            prev_option=option,
            is_forward=is_forward,
        )

        evolution_option = self.evolution_generate(
            current_chosen_mask_logits=mask_logits[0, chosen_idx], prev_option=option
        )
        return options + evolution_option

    def evolution_generate(
        self,
        current_chosen_mask_logits: torch.Tensor,
        prev_option: BeamSearchOptionData,
    ) -> List[BeamSearchOptionData]:
        if not self.allow_evolution:
            return []
        if prev_option.frame_idx is None:
            return []
        if prev_option.cyclic_count > 3:
            print("Exceed cyclic count...")
            return []

        # Evolution consider cyclic inference on itself, therefore
        # consider the previous score (not include the current score)
        prev_score = prev_option.prev_score
        previous_mask = torch.sigmoid(current_chosen_mask_logits) > 0.5
        chosen_idx, mask_logits = self.core_inference(
            idx=prev_option.frame_idx, previous_mask=previous_mask
        )
        score = self.calculate_stability_score_with_sigmoid(
            masks=mask_logits[0, chosen_idx],
            mask_threshold=0.5,
            threshold_offset=self.stability_config['offset']
        )
        
        # score = self.confidence_score(
        #     mask_logits=mask_logits[0, chosen_idx],
        #     threshold=0.5,
        # )
        score = prev_score + np.log(score + 1e-30)
        if score < prev_option.score:
            return []

        return [
            BeamSearchOptionData(
                # Skip object
                prev_id=prev_option.prev_id,
                sigmoid_threshold=0.5,
                score=score,
                prev_score=prev_score,
                frame_idx=prev_option.frame_idx,
                next_frame_idx=prev_option.next_frame_idx,
                mask_logits=mask_logits[0, chosen_idx],
                cyclic_count=prev_option.cyclic_count + 1,
            )
        ]

    def generate_next(
        self,
        current_chosen_mask_logits: torch.Tensor,
        prev_option: BeamSearchOptionData,
        is_forward=True,
    ) -> List[BeamSearchOptionData]:
        options: List[BeamSearchOptionData] = []
        prev_score = prev_option.score
        idx = prev_option.next_frame_idx
        next_frame_idx = idx + 1 if is_forward else idx - 1
        start = self.stability_config["threshold_start"]
        end = self.stability_config["threshold_end"]
        num = self.stability_config["threshold_num"]
        for value in np.linspace(start=start, stop=end, num=num):
            score = self.calculate_stability_score_with_sigmoid(
                masks=current_chosen_mask_logits,
                mask_threshold=value,
                threshold_offset=self.stability_config['offset']
                )
            # score = self.confidence_score(
            #     mask_logits=current_chosen_mask_logits,
            #     threshold=value,
            # )
            score = prev_score + np.log(score + 1e-30)
            new_option = BeamSearchOptionData(
                sigmoid_threshold=value,
                prev_id=prev_option.obj_id,
                mask_logits=current_chosen_mask_logits,
                score=score,
                prev_score=prev_score,
                frame_idx=idx,
                next_frame_idx=next_frame_idx,
            )
            options.append(new_option)
        return options

    def ranking_fn(self, options: List[BeamSearchOptionData]):
        if len(options) == 0:
            return []
        return sorted(options, key=lambda x: x.score, reverse=True)

    def seeding_strategies(self, target_idx: int, **kwargs):
        start_idx = kwargs["start_idx"]
        end_idx = kwargs["end_idx"]
        if self.strategy_name == "random-pick-one":
            init_mask = torch.as_tensor(self.masks[start_idx - 1].copy() == target_idx)
            init_mask = pick_one_pixel(
                init_mask,
                radius=int(self.start_radius),
                gaussian_config=self.gaussian_config,
                seed=self.seed,
            )
            return init_mask, start_idx

        if self.strategy_name == "local-mean-centroid":
            assert target_idx == 1, f"Only support target-idx == 1, given {target_idx}"
            proposal = make_centroid(self.masks, self.LOCAL_MEAN_CENTROID)
            z, y, x = proposal[1].tolist()
            assert (
                z > start_idx and z < end_idx
            ), f"Proposal {z=} is out of wanted range: ({start_idx}, {end_idx})"
            # Prepare the mask
            init_mask = make_gauss_point_mask(
                x=x,
                y=y,
                radius=int(self.start_radius),
                gaussian_config=self.gaussian_config,
            )
            return init_mask, z

        init_mask = torch.as_tensor(self.masks[start_idx - 1].copy() == target_idx)
        return init_mask, start_idx

    def beam_search_inference(
        self,
        start_idx: int,
        end_idx: int,
        target_idx: int,
        beam_width: int = 3,
    ):
        init_mask, proposal_start_idx = self.seeding_strategies(
            target_idx=target_idx,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        forward_data = self.beam_search(
            start_idx,
            end_idx,
            beam_width,
            proposal_start_idx,
            init_mask,
            is_forward=True,
        )
        proposal_masks = [
            forward_data[0],  # This is the middle frame
            # forward_data[-1],  # This is the last frame
        ]

        backward_data = self.beam_search(
            start_idx,
            end_idx,
            beam_width,
            proposal_start_idx,
            init_mask,
            extra_proposal_masks=proposal_masks,
            is_forward=False,
        )

        return forward_data, backward_data, init_mask, proposal_start_idx

    def beam_search(
        self,
        start_idx,
        end_idx,
        beam_width,
        proposal_start_idx,
        init_mask: np.ndarray,
        extra_proposal_masks: List[np.ndarray] = [],
        is_forward=True,
    ):
        filter_for_tracing, filter_next_round, filter_done = self.make_filter(
            start_idx, end_idx, is_forward
        )
        proposal_masks = [init_mask, *extra_proposal_masks]
        options: List[BeamSearchOptionData] = [
            BeamSearchOptionData(
                mask_logits=proposal_mask,
                frame_idx=None,
                next_frame_idx=(
                    proposal_start_idx if is_forward else proposal_start_idx - 1
                ),
            )
            for proposal_mask in proposal_masks
        ]
        tracing_tool = Tracing()
        tracing_tool.add_multi(options)
        while len(options) > 0:
            buffer: List[BeamSearchOptionData] = []
            for option in options:
                new_options = self.generate_option(option, is_forward=is_forward)
                buffer.extend(new_options)
                pass

            options = self.ranking_fn(buffer)[:beam_width]
            options = list(filter(filter_for_tracing, options))
            tracing_tool.add_multi(options)

            options = list(filter(filter_next_round, options))
            pass

        # Start Tracing
        done_option = filter(filter_done, tracing_tool.flatten())
        best_option: BeamSearchOptionData = max(done_option, key=lambda x: x.score)
        data = [best_option.get_mask()]
        opts = tracing_tool.tracing(prev_id=best_option.prev_id)
        print(list(map(lambda x: x.frame_idx, opts)))
        data.extend([x.get_mask() for x in opts])

        return data[::-1]

    def make_filter(self, start_idx, end_idx, is_forward):
        if is_forward:
            filter_for_tracing: FILTER_TYPE = (
                lambda opt: opt.next_frame_idx <= end_idx + 1
            )
            filter_next_round: FILTER_TYPE = lambda opt: opt.next_frame_idx <= end_idx
            filter_done: FILTER_TYPE = lambda opt: opt.next_frame_idx == end_idx + 1
        else:
            filter_for_tracing: FILTER_TYPE = (
                lambda opt: opt.next_frame_idx >= start_idx - 1
            )
            filter_next_round: FILTER_TYPE = lambda opt: opt.next_frame_idx >= start_idx
            filter_done: FILTER_TYPE = lambda opt: opt.next_frame_idx == start_idx - 1

        return filter_for_tracing, filter_next_round, filter_done

    def compose_result(self, backward_data, forward_data, start_idx, end_idx):
        data = np.stack([*backward_data[::-1], *forward_data])
        prediction = np.zeros(self.masks.shape)
        prediction[start_idx : end_idx + 1] = data
        return prediction


if __name__ == "__main__":
    VAL_ROOT = "./dataset/FLARE22-version1/ReleaseValGT-20cases"
    VOLUME_CACHE = os.path.join(VAL_ROOT, "images/FLARETs_0002_0000.cache.pt")
    IMAGE_PATH = os.path.join(VAL_ROOT, "images/FLARETs_0002_0000.nii.gz")
    MASK_PATH = os.path.join(VAL_ROOT, "labels/FLARETs_0002.nii.gz")
    MODEL_PATH = "./runs/transfer/imp-230603-150046/model-20.pt"
    model: Sam = sam_model_registry["vit_b"](
        checkpoint="./sam_vit_b_01ec64.pth", custom=MODEL_PATH
    )
    sam_train = SamTrain(sam_model=model)
    volumes, masks = preprocessor.run_with_config(
        image_file=IMAGE_PATH,
        gt_file=MASK_PATH,
        config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
    )
    cache_volume = torch_try_load(VOLUME_CACHE, "cpu")
    engine = BeamSearchInferenceEngine(
        volumes=volumes,
        caches=cache_volume,
        masks=masks,
        sam_train=sam_train,
        stability_config=None,
        start_radius=10,
        gaussian_config={"sigma": 10.0},
        strategy_name="local-mean-centroid",
        allow_evolution=True,
    )
    start_idx = 100
    end_idx = start_idx + 70
    (
        forward_data,
        backward_data,
        init_mask,
        proposal_start_idx,
    ) = engine.beam_search_inference(
        start_idx=start_idx,
        end_idx=end_idx,
        target_idx=1,
        beam_width=3,
    )
    print(proposal_start_idx)
    pred = engine.compose_result(backward_data, forward_data, start_idx, end_idx)
    print(pred.shape)
