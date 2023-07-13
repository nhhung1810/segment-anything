from argparse import ArgumentParser
from glob import glob
import os
import subprocess
from natsort import natsorted
from tqdm import tqdm

from scripts.datasets.constant import TEST_NON_PROCESSED

parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--model_dir",
    type=str,
    help="Model dir to be scan",
)
parser.add_argument(
    "--prefix",
    type=str,
    help="Prefix for output name",
    default='untitled',
)
parser.add_argument(
    "--limit",
    type=int,
    help="Maximal number of checkpoint to be take. Note that the checkpoint will be sort asc. Postive will take the last `limit`, Negative will take the first `limit`",
    default=None,
)

parser.add_argument(
    "--is_custom_class",
    type=bool,
    help="Is using custom class (1, 9)",
    default=False,
)
parser.add_argument(
    "--skip_eval",
    type=bool,
    help="Is skip eval?",
    default=False,
)
parser.add_argument(
    "--force_ckpt",
    type=int,
    help="Ignore the limit and only evaluate the given checkpoint",
    default=None,
)
parser.add_argument(
    "--cuda",
    type=int,
    help="CUDA index",
    default=1,
)
parser.add_argument(
    "--pid",
    type=int,
    help="PID index",
    default=None,
)

args = parser.parse_args()

if __name__ == "__main__":
    model_dir = args.model_dir
    limit = args.limit
    prefix = args.prefix
    prefix = str(prefix).lower().strip().replace(" ", "_")
    is_custom_class = args.is_custom_class
    is_skip_eval = args.skip_eval
    model = list(natsorted(list(glob(f"{model_dir}/*.pt"))))
    force_ckpt = args.force_ckpt
    if force_ckpt is None:
        if limit is not None: 
            if limit > 0:
                model = model[-limit:]
            else:
                model = model[:-limit]
    else:
        model = list(natsorted(list(glob(f"{model_dir}/model-{force_ckpt}.pt"))))
    
    for model_path in tqdm(model, total=len(model), desc="Invoke the evaluation script..."):
        model_name = os.path.basename(os.path.dirname(model_path))
        check_point = os.path.basename(model_path).replace(".pt", "").split("-")[1]
        output_dir = f"runs/submission/{model_name}/{check_point}/"
        eval_cmd = f"""
            python scripts/tools/evaluation/DSC_NSD_eval_fast.py\
                -pid {args.pid} \
                -g {TEST_NON_PROCESSED}/labels\
                -p {output_dir}\
                --name "{prefix}-{model_name}-{check_point}"
            """ if not is_skip_eval else ":"
        if not is_custom_class:
            run_inference_cmd = f"""
            python scripts/experiments/mask_aug/inference.py\
                --checkpoint {model_path} \
                --output_dir {output_dir} && {eval_cmd}
            """
        else:
            run_inference_cmd = f"""
            python scripts/experiments/mask_aug/inference.py\
                --cuda={args.cuda} \
                --selected_class 1 9 \
                --checkpoint {model_path} \
                --output_dir {output_dir} && {eval_cmd}
            """
            pass
        print(run_inference_cmd)
        subprocess.run(
            [run_inference_cmd],
            shell=True,
        )
    pass