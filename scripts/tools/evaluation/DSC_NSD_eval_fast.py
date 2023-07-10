import json
import numpy as np
import nibabel as nb
import os
import os.path as osp
from collections import OrderedDict
from SurfaceDice import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient,
)
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser("Fast evaluation, tested against the official one")
parser.add_argument(
    "-n", "--name", type=str, help="Experiment name", default="Untitled"
)
parser.add_argument(
    "-pid", "--pid", type=str, help="PID to output the log", default=None
)
parser.add_argument("-g", "--gt_dir", type=str, help="Ground Truth directory")
parser.add_argument("-p", "--pred_dir", type=str, help="Prediction directory")

NUM_CLASSES = 13


def compute(zipped_data):
    name, gt_path, seg_path = zipped_data
    seg_metrics = {}
    gt_name = name.replace("_0000.nii.gz", ".nii.gz")
    if not osp.isfile(osp.join(gt_path, gt_name)):
        return {}

    seg_metrics["Name"] = name
    seg_metrics["DSC_mean"] = 0.0
    seg_metrics["NSD-1mm_mean"] = 0.0
    # load GT truth and segmentation

    gt_nii = nb.load(osp.join(gt_path, gt_name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(osp.join(seg_path, name)).get_fdata())

    for i in range(1, NUM_CLASSES + 1):
        if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
            DSC_i = 0
            NSD_i = 0
        else:
            surface_distances = compute_surface_distances(
                gt_data == i, seg_data == i, case_spacing
            )
            DSC_i = compute_dice_coefficient(gt_data == i, seg_data == i)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)

        # Log metric
        seg_metrics[f"DSC_{i}"] = DSC_i
        seg_metrics[f"NSD-1mm_{i}"] = NSD_i

        # Accumulate mean
        seg_metrics["DSC_mean"] += DSC_i
        seg_metrics["NSD-1mm_mean"] += NSD_i
        pass

    seg_metrics["DSC_mean"] /= NUM_CLASSES
    seg_metrics["NSD-1mm_mean"] /= NUM_CLASSES
    # All of the output is in float
    return seg_metrics


def eval(args, df: pd.DataFrame, log_path: str):
    seg_path = args.pred_dir
    gt_path = args.gt_dir
    EXPERIMENT_NAME = args.name

    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith(".nii.gz")]
    filenames.sort()

    seg_metrics = OrderedDict()
    seg_metrics["Name"] = list()
    seg_metrics["DSC_mean"] = list()
    seg_metrics["NSD-1mm_mean"] = list()
    for i in range(1, NUM_CLASSES + 1):
        seg_metrics[f"DSC_{i}"] = list()
        seg_metrics[f"NSD-1mm_{i}"] = list()

    print(f"Start multiprocessing on {len(filenames)} num. of files...")
    with Pool() as pool:
        gts = [gt_path for _ in filenames]
        segs = [seg_path for _ in filenames]
        all_patient_result = pool.imap(compute, zip(filenames, gts, segs))
        for patient_result in all_patient_result:
            for i in range(1, NUM_CLASSES + 1):
                seg_metrics[f"DSC_{i}"].append(patient_result[f"DSC_{i}"])
                seg_metrics[f"NSD-1mm_{i}"].append(patient_result[f"NSD-1mm_{i}"])
            # Append the organ-mean result to a patient
            seg_metrics["DSC_mean"].append(patient_result["DSC_mean"])
            seg_metrics["NSD-1mm_mean"].append(patient_result["NSD-1mm_mean"])
            pass
    print(f"Done multiprocessing on {len(filenames)} num. of files...")

    df_dict = {}
    seg_metrics["Name"].insert(0, "All")
    seg_metrics["DSC_mean"].insert(0, np.mean(seg_metrics["DSC_mean"]))
    seg_metrics["NSD-1mm_mean"].insert(0, np.mean(seg_metrics["NSD-1mm_mean"]))

    df_dict["Name"] = EXPERIMENT_NAME
    df_dict["DSC_mean"] = seg_metrics["DSC_mean"][0]
    df_dict["NSD-1mm_mean"] = seg_metrics["NSD-1mm_mean"][0]

    for i in tqdm(range(1, NUM_CLASSES + 1), desc="Logging data..."):
        seg_metrics["DSC_{}".format(i)].insert(
            0, np.mean(seg_metrics["DSC_{}".format(i)])
        )
        seg_metrics["NSD-1mm_{}".format(i)].insert(
            0, np.mean(seg_metrics["NSD-1mm_{}".format(i)])
        )
        df_dict["DSC_{}".format(i)] = seg_metrics["DSC_{}".format(i)][0]
        df_dict["NSD-1mm_{}".format(i)] = seg_metrics["NSD-1mm_{}".format(i)][0]

    date = datetime.now()
    df2 = pd.DataFrame(
        [[date] + list(df_dict.values())], columns=["Date"] + list(df_dict.keys())
    )

    pd.concat([df, df2]).to_csv(log_path, index=False)
    with open(os.path.join(seg_path, "all-result.json"), "w") as out:
        json.dump(seg_metrics, out)
    # Print table
    # table = tabulate(seg_metrics, headers="keys", tablefmt="fancy_grid")

    # print(table)


if __name__ == "__main__":
    from time import time_ns

    args = parser.parse_args()
    pid = args.pid
    BASE_LOG = "runs/vallog.csv"
    pid_log = f"runs/vallog-{pid}.csv"
    # Construct the pid_log first
    if not os.path.exists(pid_log):
        df = pd.read_csv(BASE_LOG)
        df.to_csv(pid_log, index=False)
        pass

    df = pd.read_csv(pid_log)
    start = time_ns()
    eval(args, df, log_path=pid_log)
    stop = time_ns()
    print(f"Time elapsed: {(stop-start)/10**6}ms")
