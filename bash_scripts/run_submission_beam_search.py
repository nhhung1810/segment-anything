from argparse import ArgumentParser
from glob import glob
import itertools
import json
import os
import subprocess
from natsort import natsorted
from tqdm import tqdm
from subprocess import Popen


from scripts.datasets.constant import TEST_NON_PROCESSED

parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--model_dir",
    type=str,
    help="Model dir to be scan (won't work here...)",
)
parser.add_argument(
    "--prefix",
    type=str,
    help="Prefix for output name",
    default="untitled",
)
parser.add_argument(
    "--limit",
    type=int,
    help="Maximal number of checkpoint to be take. Note that the checkpoint will be sort asc. Positive will take the last `limit`, Negative will take the first `limit`",
    default=None,
)

parser.add_argument(
    "--skip_eval",
    type=bool,
    help="Is skip eval?",
    default=False,
)
parser.add_argument(
    "--beam_search_config_folder",
    type=str,
    help="Beam search random config folder. This config can override model-dir's scan result.",
    default=None,
)
parser.add_argument(
    "--n_processes",
    type=int,
    help="Num. of parallel processes",
    default=1,
)


args = parser.parse_args()


def scan_model_with_limit(model_dir: str, limit: int) -> list:
    model_paths = list(natsorted(list(glob(f"{model_dir}/*.pt"))))
    if limit is not None:
        if limit > 0:
            model_paths = model_paths[-limit:]
        else:
            model_paths = model_paths[:-limit]
    return model_paths


def scan_config(beam_search_config_folder: str, limit: int) -> list:
    if beam_search_config_folder is None:
        return []

    config_paths = list(natsorted(list(glob(f"{beam_search_config_folder}/*.json"))))
    if limit is not None:
        # If limit > 0 -> take from tail, limit < 0 -> take from head
        if limit > 0:
            config_paths = config_paths[-limit:]
        else:
            config_paths = config_paths[:-limit]
    return config_paths


def make_run_config(config_paths):
    if len(config_paths) == 0:
        return []

    for config_path in config_paths:
        with open(config_path, "r") as out:
            config = json.load(out)
            model_path = config["model_path"]
            hash_name = config["hash_name"]

        yield model_path, config_path, hash_name


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


if __name__ == "__main__":
    # model_dir = args.model_dir
    limit = args.limit
    # is_custom_class = args.is_custom_class
    # prefix = str(args.prefix).lower().strip().replace(" ", "_")
    is_skip_eval = args.skip_eval
    n_processes = args.n_processes

    # model_paths = scan_model_with_limit(model_dir, limit)
    config_paths = scan_config(args.beam_search_config_folder)
    run_config = make_run_config(config_paths)
    run_config_batches = batched(run_config, n_processes)
    for run_config_batch in tqdm(
        run_config_batches,
        total=len(config_paths),
        desc="Invoke the evaluation script...",
    ):
        commands = []
        for model_path, config_path, hash_name in run_config_batch:
            model_name = os.path.basename(os.path.dirname(model_path))
            check_point = os.path.basename(model_path).replace(".pt", "").split("-")[1]
            output_dir = f"runs/submission/{hash_name}"
            eval_cmd = (
                f"""
                python scripts/tools/evaluation/DSC_NSD_eval_fast.py\
                    -g {TEST_NON_PROCESSED}/labels\
                    -p {output_dir}\
                    --name "{hash_name}-{model_name}-{check_point}"
                """
                if not is_skip_eval
                else ":"
            )

            run_inference_cmd = f"""
            python scripts/experiments/beam_search/inference.py\
                --config_path {config_path} \
                --checkpoint {model_path} \
                --output_dir {output_dir} && {eval_cmd}
            """
            commands.append(run_inference_cmd)
        pass

        # commands = ['command1', 'command2']
        processes = [Popen(i, shell=True) for i in commands]
        for p in processes:
            p.wait()

        # subprocess.run(
        #     [run_inference_cmd],
        #     shell=True,
        # )
    pass
