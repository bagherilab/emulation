import os

import hydra

from config_utils import assign_models, assign_hyperparameters
from permutation.file_utils import clean_dir
from permutation.experiments.experiment import StandardExperiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    cfg = config["cs"]
    sobol_power = config["sobol_power"]
    for experiment_name in cfg.experiments:
        experiment_cfg = cfg["experiments"][experiment_name]
        for response in cfg.data.response:
            experiment = StandardExperiment(
                experiment_name=f"{experiment_name}-{response}",
                export_dir=experiment_cfg.paths.results,
                log_dir=experiment_cfg.paths.log,
                data_path=os.path.join(experiment_cfg.paths.data, experiment_cfg.files.data),
                features=cfg.data.features,
                response=response,
            )

        for model, hparam_cfg in cfg.models.items():
            temp_list = assign_hyperparameters.assign_hyperparameters(hparam_cfg, sobol_power)
            model_list = assign_models.assign_models_from_list(temp_list, model)
            experiment.add_models(model_list)

        experiment.save_manifest()
        experiment.run()


if __name__ == "__main__":
    main()
