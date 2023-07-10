from glob import glob
import json
import os
from time import time_ns
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# from scripts.constants import FLARE22_LABEL_ENUM
from scripts.datasets.constant import (
    IMAGE_TYPE,
    TEST_NON_PROCESSED,
)
from scripts.datasets.preprocess_raw import FLARE22_Preprocess
from scripts.experiments.beam_search.beam_search_engine import BeamSearchInferenceEngine
from scripts.sam_train import SamTrain
from scripts.datasets.constant import DEFAULT_DEVICE

from scripts.tools.evaluation.loading import post_process

# from scripts.tools.profiling import GPUProfiler
from scripts.utils import make_directory, omit, torch_try_load, pick
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam
from scripts.losses.loss import DiceLoss
from argparse import ArgumentParser

from loguru import logger


DICE_FN = DiceLoss(activation=None, reduction="none")

logger.remove()
logger.add(
    "inference-log.txt",
    format="<lvl>[{time:DD:MMM:YY HH:mm:ss}] - [{level}] - {message}</lvl>",
)


def load_model(model_path, device=DEFAULT_DEVICE) -> Sam:
    model: Sam = sam_model_registry["vit_b"](
        checkpoint="./sam_vit_b_01ec64.pth", custom=model_path
    )
    model.to(device)
    return model


def inference(
    images: List[str],
    gts: List[str],
    sam_train: SamTrain,
    inference_save_dir: str,
    device: str,
    use_cache: bool,
    beam_search_config_path: str = None,
):
    assert len(images) == len(gts)
    preprocessor = FLARE22_Preprocess()
    total_times = []
    if beam_search_config_path is not None:
        assert os.path.exists(beam_search_config_path)
        with open(beam_search_config_path, "r") as out:
            beam_search_config: Dict = json.load(beam_search_config_path)
    else:
        # TODO: add default
        beam_search_config: Dict = BeamSearchInferenceEngine.make_default_config()

    # Only take what it need
    beam_search_config = pick(
        beam_search_config,
        keys=list(BeamSearchInferenceEngine.make_default_config().keys()),
    )

    for image_file, gt_file in tqdm(
        zip(images, gts), total=len(images), desc="Inference for patient..."
    ):
        stime = time_ns()
        cache_volume_path = image_file.replace(".nii.gz", ".cache.pt")
        volumes, masks = preprocessor.run_with_config(
            image_file=image_file,
            gt_file=gt_file,
            config_name=IMAGE_TYPE.ABDOMEN_SOFT_TISSUES_ABDOMEN_LIVER,
        )

        cache_volume = torch_try_load(cache_volume_path, device=device)
        engine = BeamSearchInferenceEngine(
            volumes=volumes,
            caches=cache_volume,
            masks=masks,
            sam_train=sam_train,
            **beam_search_config,
        )

        start_idx = engine.starts[1]
        end_idx = engine.ends[1]
        fwd_data, bwd_data, _, _ = engine.beam_search_inference(
            start_idx=start_idx,
            end_idx=end_idx,
            target_idx=1,
            beam_width=3,
        )
        predict_volume = engine.compose_result(bwd_data, fwd_data, start_idx, end_idx)

        predict_volume = predict_volume.transpose(1, 2, 0).astype(np.uint8)
        etime = time_ns()
        total_times.append(etime - stime)
        post_process(
            pred=predict_volume,
            gt_file=gt_file,
            out_dir=inference_save_dir,
        )
        pass
    return total_times


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
    default=False,
)
parser.add_argument(
    "--beam_search_config_path",
    type=str,
    help="Algorithm config path to be loaded",
    default=None,
)

parser.add_argument(
    "--selected_class",
    nargs="+",
    type=int,
    help="List of class, no pre/pos-fix separated by space, i.e. 1 2 3",
    default=None,  # for liver and gallbladder
)


def make_input_path(input_dir, label_dir):
    images_path: List[str] = sorted(glob(f"{input_dir}/*.nii.gz"))
    images_path = [os.path.basename(p) for p in images_path]
    # By some way, they don't have gallbladder,
    # which i will omit for now
    val_remove = [
        f"FLARETs_00{sym}_0000.nii.gz"
        for sym in ["06", "08", "21", "31", "33", "36", "38", "43", "44", "48"]
    ]
    images_path = [img for img in images_path if img not in val_remove]
    # images_path = [img for img in images_path if img == f"FLARETs_0002_0000.nii.gz"]
    labels_path = [
        os.path.join(label_dir, p.replace("_0000.nii.gz", ".nii.gz"))
        for p in images_path
    ]
    images_path = [os.path.join(input_dir, p) for p in images_path]

    for i, p in zip(images_path, labels_path):
        assert os.path.exists(i), f"{i}"
        assert os.path.exists(p), f"{p}"
        pass
    return images_path, labels_path


if __name__ == "__main__":
    args = parser.parse_args()
    device = f"cuda:{args.cuda}" if "cuda" in DEFAULT_DEVICE else DEFAULT_DEVICE

    run_path = "mask-prop-230509-005503"
    model_path = args.checkpoint or f"runs/{run_path}/model-100.pt"
    # model_path = "./runs/transfer/imp-230603-150046/model-20.pt"
    model = load_model(model_path, device)
    sam_train = SamTrain(sam_model=model)

    inference_save_dir = args.output_dir
    input_dir = args.input_dir
    label_dir = args.label_dir
    use_cache = args.use_cache
    beam_search_config_path = args.beam_search_config_path
    selected_class = [1]
    print(
        f"""
        model-path     : {model_path}
        save-dir       : {inference_save_dir}
        input-dir      : {input_dir}
        label-dir      : {label_dir}
        is-use-cache   : {use_cache}
        selected-class : {selected_class}
        bs-config-path : {beam_search_config_path}
        """
    )
    make_directory(inference_save_dir)

    images_path, labels_path = make_input_path(input_dir, label_dir)
    total_times = inference(
        images=images_path,
        gts=labels_path,
        inference_save_dir=inference_save_dir,
        sam_train=sam_train,
        device=device,
        use_cache=use_cache,
        beam_search_config_path=beam_search_config_path,
    )
    logger.success(f"Mean time: {np.mean(total_times)}")

    pass
