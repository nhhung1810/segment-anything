from argparse import ArgumentParser
from datetime import datetime
from glob import glob
import json
import os
import pandas as pd

from scripts.utils import pick

TIME = datetime.now().strftime("%y%m%d-%H%M%S")

parser = ArgumentParser(
    "Export a csv file of all experiment config. Adapt for 'sacred', a python library for experiment tracking"
)
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="Directory to scan for config",
    default="runs",
)
parser.add_argument(
    "-d",
    "--depth",
    type=int,
    help="Depth of scanning",
    default=2,
)
parser.add_argument(
    "-o", "--output", type=str, help="Output csv file", default=f"config-{TIME}.csv"
)


def load_json(path):
    with open(path, "r") as out:
        d = json.load(out)

    return d

def scan(input_dir, depth=1, exludes=[]):
    scan_depth = "*/" * depth
    glob_string = f"{input_dir}/{scan_depth}/_sources/"
    sacred_paths = list(glob(glob_string, recursive=True))
    sacred_paths = [p.replace("_sources/", "") for p in sacred_paths]
    all_dfs = []
    for experiment_folder in sacred_paths:
        df_dict = {}
        config_path = os.path.join(experiment_folder, "1/config.json")
        runs_path = os.path.join(experiment_folder, "1/run.json")
        num_checkpoint = len(glob(f"{experiment_folder}/*.pt"))

        config_data = load_json(config_path)
        run_metadata = load_json(runs_path)
        experiment_main_name = run_metadata["experiment"]["name"]
        if experiment_main_name in exludes:
            continue
        df_dict = {
            "name": experiment_main_name,
            "n_checkpoint": num_checkpoint,
            **pick(run_metadata, ["start_time", "stop_time", "status"]),
            **pick(
                config_data,
                [
                    "batch_size",
                    "gradient_accumulation_step",
                    "logdir",
                    "custom_model_path",
                    "class_selected",
                    "n_epochs",
                    "evaluate_epoch",
                    "save_epoch",
                    "learning_rate",
                    "learning_rate_decay_steps",
                    "learning_rate_decay_rate",
                    "learning_rate_decay_patience",
                    "aug_dict",
                ],
            ),
        }

        temp_df = pd.DataFrame([list(df_dict.values())], columns=list(df_dict.keys()))
        all_dfs.append(temp_df)
    return all_dfs


if __name__ == "__main__":
    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output
    depth = args.depth
    exludes = [
        'sam-fix-iou',
        'sam-one-point',
        'organ-ctx'
    ]
    all_dfs = []
    for d in range(1, depth + 1, 1):
        all_dfs.extend(scan(input_dir, depth=d, exludes=exludes))
    pd.concat(all_dfs).to_csv(output_file, index=False)
    pass
