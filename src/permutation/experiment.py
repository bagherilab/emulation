from typing import Optional

import numpy as np

from permutation.runner import Runner
from permutation.stage import Stage, IncorrectStageException
from permutation.loader import Loader, CSVLoader
from permutation.logger import Logger
from permutation.models.modelprotocol import Model


class Experiment:
    """todo"""

    def __init__(
        self, experiment_name: str, log_dir: str, data_path: str, features: list[str], response: str
    ) -> None:
        """todo"""

        self.name = experiment_name
        self.logger: Logger = Logger(log_dir)
        self.loader: Loader = CSVLoader(data_path, features, response)
        self._features = features
        self._response = response
        self._models: dict[str, list[Runner]] = {}
        self._best_models: dict[str, Runner] = {}

    def add_model(self, model: Model) -> None:
        """todo"""
        self._check_algorithm_in_models(model.algorithm_name)
        self._models[model.algorithm_name].append(Runner(model, self.loader))

    def _check_algorithm_in_models(self, name: str) -> None:
        """todo"""
        if name not in self._models:
            self._models[name] = []

    @property
    def models(self) -> list[str]:
        """todo"""
        list_of_models = []
        for _, runner_list in self._models.items():
            temp_name_list = [runner.name for runner in runner_list]
            list_of_models.extend(temp_name_list)
        return list_of_models

    @property
    def algorithms(self) -> list[str]:
        """todo"""
        return list(self._models.keys())

    def hyperparameter_selection(self) -> None:
        """todo"""
        self._get_best_cv_model()

    def _get_best_cv_model(self) -> None:
        """todo"""
        for algorithm, runner_list in self._models.items():
            map(lambda r: r.cross_validation(), runner_list)
            # if None, an IncorrectStageException will be thrown before the next line runs
            cv_list = [runner.cv_metrics.average for runner in runner_list]  # type: ignore
            best_value = max(cv_list)
            model_index = next(i for i, val in enumerate(cv_list) if val == best_value)
            self._update_best_model(algorithm, model_index)
            self._log_hyperparameter_selection(algorithm)

    def _update_best_model(self, algorithm: str, model_index: int) -> None:
        """todo"""
        self._best_models[algorithm] = self._models[algorithm][model_index]
        self._best_models[algorithm].set_stage(Stage.TRAIN)

    def _log_hyperparameter_selection(self, algorithm: str) -> None:
        """todo"""
        for runner in self._models[algorithm]:
            # if None, an IncorrectStageException will be thrown before the next line runs
            self.logger.metric_to_csv(
                self.name, algorithm, runner.name, runner.cv_metrics  # type: ignore
            )

    def train_models(self) -> None:
        """todo"""
        for algorithm, runner in self._best_models.items():
            runner.train()
            runner.set_stage(Stage.TEST)
            self._log_training_performance(algorithm)

    def _log_training_performance(self, algorithm: str) -> None:
        """todo"""
        runner = self._best_models[algorithm]
        self.logger.metric_to_csv(self.name, algorithm, runner.name, runner.training_metrics)

    def test_models(self) -> None:
        """todo"""
        for algorithm, runner in self._best_models.items():
            runner.test()
            runner.set_stage(Stage.PERM)
            self._log_test_performance(algorithm)

    def _log_test_performance(self, algorithm: str) -> None:
        """todo"""
        runner = self._best_models[algorithm]
        self.logger.metric_to_csv(self.name, algorithm, runner.name, runner.testing_metrics)

    def permutation_testing(self) -> None:
        """todo"""
        for algorithm, runner in self._best_models.items():
            runner.permutation_testing()
            self._log_perm_performance(algorithm)

    def _log_perm_performance(self, algorithm: str) -> None:
        """todo"""
        runner = self._best_models[algorithm]
        for perm_metric in runner.permutation_metrics:
            self.logger.metric_to_csv(
                self.name, algorithm, f"{runner.name}_{perm_metric.name}", perm_metric
            )

    def run_standard_experiment(self) -> None:
        """
        Run the member methods associated with the experiment.
        1) Hyperparameter selection via cross validation.
        2) Model training of the best models selected in (1)
        3) Model performance testing of the best models selected in (1)
        4) Permutation testing of the best models selected in (1)
        """
        self._reset_best_models()
        try:
            self.hyperparameter_selection()
        except IncorrectStageException:
            # No hyperparameters were passed TODO: add logger functionality to log to file
            pass
        self.train_models()
        self.test_models()
        self.permutation_testing()

    def run_training_quantity_experiment(
        self, num: int = 10, repeat: Optional[int] = False
    ) -> None:
        """
        Run the member methods associated with the experiment.
        Repeats standard experiment N times
        """
        evenly_spaced_arr = np.linspace(0, self.loader.n_total, num=num, dtype=int)
        for val in evenly_spaced_arr:
            if not repeat:
                self._subsample_and_run(n=val)
            else:
                self._run_repeats(n=val, repeats=repeat)

    def _subsample_and_run(self, n: int) -> None:
        """todo"""
        self.loader.subsample(n)
        self.run_standard_experiment()

    def _run_repeats(self, n: int, repeats: int) -> None:
        """
        helper function for training quantity experiment,
        handles naming and rerunning experiment
        """
        original = self.name
        for i in range(repeats):
            temp_name = f"{original}_{i}"
            self.name = temp_name
            self._subsample_and_run(n=n)
        self.name = original

    def _reset_best_models(self) -> None:
        """todo"""
        self._best_models = {}
