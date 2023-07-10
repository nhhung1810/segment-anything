import json
import os
from typing import Dict
from uuid import uuid4
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import expon

from scripts.utils import make_directory, omit


class RandomSearchGenerator:
    def __init__(self) -> None:
        self.fix_config = {
            "seed": None,
            "strategy_name": "local-mean-centroid",
            "allow_evolution": True,
        }
        self.config = {
            "stability_config": [True, False],
            "stability_config.threshold_start": [0.1, 0.3],
            "stability_config.threshold_stop": [0.7, 0.9],
            "stability_config.threshold_num": [10, 15, 20, 25],
            "start_radius": [3, 10, 20, 40, 60, 100, 125, 150],
            "gaussian_config.sigma": [3.0, 5.0, 10.0, 15.0, 20.0, 50.0],
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

    def generate_and_save(self, model_path: str, config_dir: str, n_iter=10):
        make_directory(config_dir)
        for config in self.generate(model_path, n_iter):
            hash_name = config["hash_name"]
            with open(os.path.join(config_dir, hash_name), "w") as out:
                json.dump(config, out)
                pass

    def add_models(self):
        pass

    # def add(self, key, )


if __name__ == "__main__":
    generator = RandomSearchGenerator()
    for config in generator.generate(model_path=""):
        print(config)
    pass
