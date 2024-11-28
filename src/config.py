import os

import hydra
from omegaconf.dictconfig import DictConfig

from config_utils import assign_models, assign_hyperparameters
from permutation.experiments.experiment import (
    StandardExperiment,
    TrainingQuantityExperiment,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config["cs"]
    sobol_power = config["sobol_power"]
    stratify = config["stratify"]
    quantity_experiment = config["quantity_experiment"]
    clean_data = config["clean_data"]

    for experiment_name in cfg.experiments:
        experiment_cfg = cfg["experiments"][experiment_name]
        for response in cfg.data.response:
            if not quantity_experiment:
                experiment = StandardExperiment(
                    experiment_name=f"{experiment_name}-{response}",
                    export_dir=experiment_cfg.paths.results,
                    log_dir=experiment_cfg.paths.log,
                    data_path=os.path.join(experiment_cfg.paths.data, experiment_cfg.files.data),
                    features=cfg.data.features,
                    response=response,
                    stratify=stratify,
                    clean_data_flag=clean_data,
                )
            else:
                experiment = TrainingQuantityExperiment(  # type: ignore
                    experiment_name=f"{experiment_name}-{response}",
                    export_dir=experiment_cfg.paths.results,
                    log_dir=experiment_cfg.paths.log,
                    data_path=os.path.join(experiment_cfg.paths.data, experiment_cfg.files.data),
                    features=cfg.data.features,
                    response=response,
                    stratify=stratify,
                    clean_data_flag=clean_data,
                )

            for model, hparam_cfg in cfg.models.items():
                temp_list = assign_hyperparameters.assign_hyperparameters(hparam_cfg, sobol_power)
                model_list = assign_models.assign_models_from_list(temp_list, model)
                experiment.add_models(model_list)

            experiment.save_train_test()
            experiment.save_manifest()
            experiment.run()
            """
            model = experiment._best_models['MDN'].model.model
            test_points = [-1, 0, 1]
            print("\nExample predictions at specific points:")
            import torch
            for x_val in test_points:
                x = torch.tensor([[x_val]], dtype=torch.float32)
                with torch.no_grad():
                    pi, mu, sigma = model.forward(x)
                print(f"\nx = {x_val}")
                print(f"Mixing coefficients (π): {pi.numpy().flatten()}")
                print(f"Means (μ): {mu.numpy().flatten()}")
                print(f"Standard deviations (σ): {sigma.numpy().flatten()}")
            """
if __name__ == "__main__":
    main()
