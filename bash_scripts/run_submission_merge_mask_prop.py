from argparse import ArgumentParser
from glob import glob
import os
import subprocess
from natsort import natsorted



parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--model_dir",
    type=str,
    help="Model dir to be scan",
)
parser.add_argument(
    "--limit",
    type=int,
    help="Maximal number of checkpoint to be take. Note that the checkpoint will be sort asc",
    default=10,
)
args = parser.parse_args()

if __name__ == "__main__":
    model_dir = args.model_dir
    limit = args.limit
    model = list(natsorted(list(glob(f"{model_dir}/*.pt"))))[:limit]
    for model_path in model:
        model_name = os.path.basename(os.path.dirname(model_path))
        check_point = os.path.basename(model_path).replace(".pt", "").split("-")[1]
        output_dir = f"runs/submission/{model_name}/{check_point}/"
        run_inference_cmd = f"""
        python scripts/experiments/simple_mask_propagate/inference_merge_class.py\
            --checkpoint={model_path} \
            --output_dir={output_dir} && \
        python scripts/tools/evaluation/DSC_NSD_eval.py\
            -g=dataset/FLARE22-version1/FLARE22_LabeledCase50/labels\
            -p={output_dir}\
            --name="merge-min-area-{model_name}-{check_point}"
        """
        subprocess.run(
            [run_inference_cmd],
            shell=True,
        )
    pass