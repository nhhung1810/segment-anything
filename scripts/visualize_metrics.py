from collections import OrderedDict
import json

from natsort import natsorted
from scripts.utils import omit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sort(metrics: dict) -> OrderedDict:
    # Sort the metrics in order: [pretrain, model-x, ...] with x as integer
    result = OrderedDict()
    result["pretrain"] = metrics["pretrain"]
    for k in natsorted(omit(metrics, ["pretrain"]).keys()):
        result[k] = metrics[k]
    return result


def extract_index(model_name: str):
    if model_name == "pretrain":
        return 0
    return int(model_name.replace("model-", "").replace(".pt", ""))


def visualize(metrics: dict):
    metrics = sort(metrics)
    for model_name, metric in metrics.items():
        index = extract_index(model_name)
        pass
    pass


def to_csv(metrics, path):
    metrics = sort(metrics)
    header = ["name", "dice/mean_of_best", "dice/mean"]
    data = []
    for model_name, metric in metrics.items():
        # index = extract_index(model_name)
        _data = [model_name]
        for k in header[1:]:
            _data.append(metric[k])
            pass
        data.append(_data)
        pass

    df = pd.DataFrame(data, columns=header)
    df.to_csv(path)

    pass


if __name__ == "__main__":
    path = "sam-one-point-230501-152140-one-point.json"
    # path = "runs/sam-fix-iou-230429-011855-one-point.json"
    with open(path, "r") as out:
        metrics = json.load(out)

    to_csv(metrics, path.replace(".json", ".csv"))
    pass
