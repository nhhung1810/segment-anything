from copy import deepcopy
from glob import glob
import json
import os
import shutil
from typing import Dict, List
from uuid import uuid4
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import expon, uniform

from scripts.utils import make_directory, omit

def load_json(path):
    with open(path, "r") as out:
        d = json.load(out)

    return d

def flatten(dictionary: Dict[str, object]):
    result = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            subresult = flatten(v)
            for sub_k, sub_v in subresult.items():
                result[f"{k}.{sub_k}"] = sub_v
        else:
            result[k] = v
        
        pass
    return result

# round_1_config = {
#     "stability_config": [True, False],
#     "stability_config.threshold_start": [0.05, 0.2, 0.4],
#     "stability_config.threshold_stop": [0.6, 0.8, 0.9],
#     "stability_config.threshold_num": [10, 15, 20, 25],
#     "start_radius": uniform(loc=1, scale=200, ),
#     # gaussian kernel size -> radius = round(4.0 * sigma)
#     "gaussian_config.sigma": [3.0, 5.0, 10.0, 15.0, 20.0, 50.0],
# }

# round_2_config = {
#     "stability_config": [True],
#     "stability_config.threshold_start": [0.4],
#     "stability_config.threshold_stop": [0.9],
#     "stability_config.threshold_num": [10, 20],
#     "start_radius": uniform(loc=60, scale=100), # loc -> loc + scale
#     # gaussian kernel size -> radius = round(4.0 * sigma)
#     "gaussian_config.sigma": uniform(loc=1, scale=40),
# }

class RandomSearchGenerator:
    def __init__(self) -> None:
        self.fix_config = {
            "seed": None,
            "strategy_name": "local-mean-centroid",
            "allow_evolution": True,
        }
        self.config = {
            "stability_config": [True],
            "stability_config.threshold_start": [0.4],
            "stability_config.threshold_stop": [0.9],
            "stability_config.threshold_num": [10],
            "start_radius": [120],
            # gaussian kernel size -> radius = round(4.0 * sigma)
            "gaussian_config.sigma": [21.0], # hyper-search result
        }
        pass

    def parse_other(self, config: Dict[str, object]):
        other_config = omit(
            config,
            [
                k
                for k in config.keys()
                if k.startswith("stability_config") or k.startswith("gaussian_config")
            ],
        )
        return {
            k: v for k, v in other_config.items() if not k.startswith("gaussian_config")
        }

    def parse_gaussian(self, config: Dict[str, object]):
        return {
            "gaussian_config": {
                k.replace("gaussian_config.", ""): v
                for k, v in config.items()
                if k.startswith("gaussian_config")
            }
        }

    def parse_stability_config(self, config: Dict[str, object]):
        if not config["stability_config"]:
            return {"stability_config": None}

        return {
            "stability_config": {
                k.replace("stability_config.", ""): v
                for k, v in config.items()
                if k.startswith("stability_config")
            }
        }

    def generate(self, model_path: str, n_iter=10):
        sampler = ParameterSampler(param_distributions=self.config, n_iter=n_iter)
        param_list = list(sampler)
        param_dict = [dict((k, v) for (k, v) in d.items()) for d in param_list]
        for config in param_dict:
            try:
                hash_name = uuid4().hex
                parsed_config = {
                    **self.parse_gaussian(config),
                    **self.parse_other(config),
                    **self.parse_stability_config(config),
                    **self.fix_config,
                    "hash_name": hash_name,
                    "model_path": model_path,
                }
                yield parsed_config
            except Exception as msg:
                print(msg)
                continue

    def generate_and_save(self, model_path: str, config_dir: str, n_iter=10):
        make_directory(config_dir)
        for config in self.generate(model_path, n_iter):
            if config is None: continue
            hash_name = config["hash_name"]
            with open(os.path.join(config_dir, f"{hash_name}.json"), "w") as out:
                json.dump(config, out)
                pass

    def generate_model_and_save(self, model_paths: List[str], config_dir: str):
        make_directory(config_dir)
        config = next(self.generate('', n_iter=1))
        for model_path in model_paths:
            if config is None: continue
            new_config = deepcopy(config)
            new_config["hash_name"] = uuid4().hex
            new_config['model_path'] = model_path
            hash_name = new_config["hash_name"]
            
            with open(os.path.join(config_dir, f"{hash_name}.json"), "w") as out:
                json.dump(new_config, out)
                pass

    def collect_beam_search_result(self, config_dir):
        all_result = []
        # Collect hash-names
        submission_dir = "runs/submission/beam-data"
        hash_names = os.listdir(config_dir)
        try:
            hash_names.remove('merge-result.json')
        except Exception as msg:
            print(msg)
        
        # print(hash_names)
        for hash_name in hash_names:
            hash_name = hash_name.replace(".json", "")
            result_json = os.path.join(submission_dir, hash_name, "all-result.json")
            config_json = os.path.join(submission_dir, hash_name, "beam-config.json")
            try:
                result = load_json(result_json)
                config = load_json(config_json)
                config = flatten(config)
                mean_eval = result['DSC_1'][0]
                config['mean_eval_liver'] = mean_eval
                all_result.append(config)
            except Exception as msg:
                print(msg)
                continue
            
        output_path = f"{config_dir}/merge-result.json"
        
        with open(output_path, 'w') as out:
            json.dump(all_result, out)
            pass

        return output_path




if __name__ == "__main__":
    generator = RandomSearchGenerator()
    # patterns = [
    #     "runs/imp-230603-150046/model-*.pt",
    #     "runs/imp-230608-231031/model-*.pt",
    #     "runs/imp-230610-011507/model-*.pt",
    #     "runs/imp-aug-230605-000452/model-*.pt",
    #     "runs/imp-aug-230605-165716/model-*.pt",
    #     "runs/imp-aug-230606-002414/model-*.pt",
    #     "runs/imp-aug-230608-003029/model-*.pt",
    #     "runs/imp-aug-230610-104249/model-*.pt",
    #     "runs/imp-aug-230610-211354/model-*.pt",
    # ]
    # model_paths = [
    #     p for pattern in patterns for p in list(glob(pattern))
    # ]
    model_paths = ['runs/imp-aug-230605-165716/model-35.pt']
    print(len(model_paths))
    
    config_dir = "runs/beam-seach-cfd/"
    # output_path = generator.collect_beam_search_result(config_dir)
    # print(f"Output path: {output_path}")
    generator.generate_model_and_save(model_paths=model_paths, config_dir=config_dir)
    print(f"Paste this into the run submission: {config_dir}")
    pass
