import os

import hydra

from config_utils import assign_models, assign_hyperparameters
from permutation.file_utils import clean_dir
from permutation.experiments.experiment import StandardExperiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    cfg = config["cs"]

    for response in cfg.data.response:
        experiment = StandardExperiment(
            experiment_name=f"{cfg.experiment.name}-{response}",
            export_dir=cfg.paths.results,
            log_dir=cfg.paths.log,
            data_path=os.path.join(cfg.paths.data, cfg.files.data),
            features=cfg.data.features,
            response=response,
        )

        for model, hparam_cfg in cfg.models.items():
            temp_list = assign_hyperparameters.assign_hyperparameters(hparam_cfg)
            model_list = assign_models.assign_models_from_list(temp_list, model)
            experiment.add_models(model_list)

        experiment.save_manifest()
        experiment.run()


if __name__ == "__main__":
    main()


# {
#     "models": {
#         "mlr": {
#             "hyperparameters": {
#                 "continuous": {
#                     "alpha": {"type": "float", "range": [0.0001, 1.0]},
#                     "l1_ratio": {"type": "float", "range": [0.0, 1.0]},
#                 }
#             }
#         },
#         "rf": {
#             "hyperparameters": {
#                 "continuous": {
#                     "n_estimators": {"type": "int", "range": [1, 100]},
#                     "max_features": {"type": "float", "range": [0.0, 1.0]},
#                     "max_depth": {"type": "int", "range": [1, 100]},
#                     "min_samples_split": {"type": "float", "range": [0.0, 1.0]},
#                     "min_samples_leaf": {"type": "float", "range": [0.0, 1.0]},
#                 },
#                 "discrete": {"bootstrap": [True, False]},
#             }
#         },
#         "svr": {
#             "hyperparameters": {
#                 "continuous": {
#                     "C": {"type": "float", "range": [0.0001, 1.0]},
#                     "epsilon": {"type": "float", "range": [0.0, 1.0]},
#                 },
#                 "discrete": {"kernel": ["linear", "poly", "rbf", "sigmoid"]},
#             }
#         },
#         "mlp": {
#             "hyperparameters": {
#                 "continuous": {"alpha": {"type": "float", "range": [0.0001, 1]}},
#                 "discrete": {
#                     "activation": ["identity", "logistic", "tanh", "relu"],
#                     "hidden_layer_sizes": [
#                         "(25,)",
#                         "(25,25)",
#                         "(50,)",
#                         "(50, 50)",
#                         "(50,25)",
#                         "(100,)",
#                         "(100,100)",
#                         "(100,50)",
#                     ],
#                 },
#                 "static": {"solver": "lbfgs"},
#             }
#         },
#     },
#     "experiment": {"name": "arcade-case-study", "debug": False},
#     "files": {"data": "arcade_emulation.csv"},
#     "paths": {
#         "log": "${hydra:runtime.cwd}/logs/",
#         "data": "${hydra:runtime.cwd}/data/",
#         "results": "${hydra:runtime.cwd}/results/",
#     },
#     "data": {
#         "features": [
#             "seed",
#             "edges",
#             "nodes",
#             "gradius",
#             "gdiameter",
#             "indegree",
#             "outdegree",
#             "degree",
#             "eccentricity",
#             "shortpath",
#             "clustering",
#             "closeness",
#             "betweenness",
#             "components",
#             "context",
#             "vasculature",
#         ],
#         "response": ["activity", "growth", "cycles", "symmetry"],
#     },
#     "debug": True,
# }
