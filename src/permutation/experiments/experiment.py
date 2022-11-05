from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from permutation.runner import Runner
from permutation.stage import Stage, IncorrectStageException
from permutation.loader import Loader, CSVLoader
from permutation.logger import Logger, ExperimentLogger
from permutation.exporter import Exporter
from permutation.models.modelprotocol import Model


class Experiment(ABC):
    name: str
    exporter: Exporter
    logger: Logger
    loader: Loader

    @abstractmethod
    def add_model(self, model: Model) -> None:
        ...

    @abstractmethod
    def add_models(self, models: list[Model]) -> None:
        ...

    @abstractmethod
    def run(self) -> None:
        ...

    @property
    @abstractmethod
    def models(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def algorithms(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def best_models(self) -> dict[str, str]:
        ...


class StandardExperiment(Experiment):
    """

    Arguments
    ----------
    experiment_name :
        name of experiment for file management purposes
    log_dir :
        directory to save experiment results
    data_path :
        location of original datafile
    features :
        list of features in the dataset to use to train models
    response :
        response variable to use as the target variable for training

    Attributes
    ----------
    name :
        location where files will be stored
    exporter :
        Exporter object where the experiment will export results
    logger :
        Logger object where the experiment will log status
    loader :
        Loader object where the experiment will load data
    models:
        models that have been added to the experiment
    algorithms :
        algorithms that have been added to the experiment
    best_models :
        hyperparameter models that have been identified as the best model through CV

    Methods
    -------
    hyperparameter_selection():
        Perform hyperparameter selection on all models for all algorithms
        Add models to best_models

    train_models():
        Train models in best_models

    test_models():
        train_models() must be run first.
        Get performance of models in best_models

    permutation_testing():
        train_models() must be run first.
        Get permutation testing results of all models in _best_models

    run_standard_experiment():
        Run the member methods associated with standard ML practices.
        1) Hyperparameter selection via cross validation.
        2) Model training of the best models selected in (1)
        3) Model performance testing of the best models selected in (1)
        4) Permutation testing of the best models selected in (1)

    run_training_quantity_experiment(n, repeats):
        Run the standard experiment different amounts of training data available.
        n: number of quantity experiments
        repeats: replicates of the subsample (account for noise in dataset)

    """

    def __init__(
        self,
        experiment_name: str,
        export_dir: str,
        log_dir: str,
        data_path: str,
        features: list[str],
        response: str,
    ) -> None:
        self.name = experiment_name
        self.exporter: Exporter = Exporter(self.name, export_dir)
        self.logger: Logger = ExperimentLogger(self.name, log_dir)
        self.loader: Loader = CSVLoader(data_path, features, response)
        self._features = features
        self._response = response
        self._models: dict[str, list[Runner]] = {}
        self._model_ids: set = set()
        self._n_models = 0
        self._best_models: dict[str, Runner] = {}
        self._log_initialization()

    def _log_initialization(self):
        self.logger.log(f"{self.name} initialized.")

    def add_model(self, model: Model) -> None:
        """todo"""
        self._check_algorithm_in_models(model.algorithm_name)
        runner = Runner(model, self.loader)
        self._models[model.algorithm_name].append(runner)
        model_id = self._check_ids(runner)
        self.logger.log(f"Added {model_id}.")
        self._n_models += 1

    def _check_ids(self, runner: Runner) -> str:
        """todo"""
        while runner.id in self._model_ids:
            self.logger.log(f"Reset id for {runner.Description}")
            runner.reset_id()

        return runner.id

    def add_models(self, models: list[Model]) -> None:
        """todo"""
        for model in models:
            self.add_model(model)

    def _check_algorithm_in_models(self, name: str) -> None:
        """todo"""
        if name not in self._models:
            self._models[name] = []

    @property
    def models(self) -> list[str]:
        """todo"""
        list_of_models = []
        for _, runner_list in self._models.items():
            temp_name_list = [runner.id for runner in runner_list]
            list_of_models.extend(temp_name_list)
        return list_of_models

    @property
    def algorithms(self) -> list[str]:
        """returns algorithms included in the experiment"""
        return list(self._models.keys())

    @property
    def best_models(self) -> dict[str, str]:
        """returns dictionary and name of best models"""
        return {alg: runner.id for alg, runner in self._best_models.items()}

    def hyperparameter_selection(self) -> None:
        """perform hparam selection using cv"""
        self._get_best_cv_model()

    def _get_best_cv_model(self) -> None:
        """run CV for all models and identify best model"""
        progress_counter = 0

        for algorithm, runner_list in self._models.items():
            for r in runner_list:
                r.cross_validation()
                progress_counter += 1
                self.logger.log(f"Progress: {progress_counter} of {self._n_models}")

            cv_list = [runner.cv_metrics.average for runner in runner_list]  # type: ignore
            best_value = max(cv_list)
            model_index = next(i for i, val in enumerate(cv_list) if val == best_value)
            self._update_best_model(algorithm, model_index)
            self._log_hyperparameter_selection(algorithm)

    def _update_best_model(self, algorithm: str, model_index: int) -> None:
        """update best_model"""
        self._best_models[algorithm] = self._models[algorithm][model_index]
        self._best_models[algorithm].set_stage(Stage.TRAIN)

    def _log_hyperparameter_selection(self, algorithm: str) -> None:
        """log the data from cv"""
        for runner in self._models[algorithm]:
            # if None, an IncorrectStageException will be thrown before the next line runs
            self.exporter.metric_to_csv(
                self.name, algorithm, runner.id, runner.cv_metrics  # type: ignore
            )

    def train_models(self) -> None:
        """train models in best_model"""
        for algorithm, runner in self._best_models.items():
            runner.train()
            runner.set_stage(Stage.TEST)
            self._log_training_performance(algorithm)

    def _log_training_performance(self, algorithm: str) -> None:
        """log the training performance"""
        runner = self._best_models[algorithm]
        self.exporter.metric_to_csv(self.name, algorithm, runner.id, runner.training_metrics)

    def test_models(self) -> None:
        """test trained best_model performances"""
        for algorithm, runner in self._best_models.items():
            runner.test()
            runner.set_stage(Stage.PERM)
            self._log_test_performance(algorithm)

    def _log_test_performance(self, algorithm: str) -> None:
        """log test performance"""
        runner = self._best_models[algorithm]
        self.exporter.metric_to_csv(self.name, algorithm, runner.id, runner.testing_metrics)

    def permutation_testing(self) -> None:
        """perform permutation testing"""
        for algorithm, runner in self._best_models.items():
            runner.permutation_testing()
            self._log_perm_performance(algorithm)

    def _log_perm_performance(self, algorithm: str) -> None:
        """log the permutation performance"""
        runner = self._best_models[algorithm]
        for perm_metric in runner.permutation_metrics:
            self.exporter.metric_to_csv(
                self.name, algorithm, f"{runner.id}_{perm_metric.name}", perm_metric
            )

    def run(self):
        self._run_standard_experiment()

    def _run_standard_experiment(self) -> None:
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

    def run_training_quantity_experiment(self, num: int = 10, repeat: Optional[int] = None) -> None:
        """
        Run the member methods associated with the experiment.
        Repeats standard experiment times
        """
        evenly_spaced_arr = np.linspace(0, self.loader.n_total, num=num, dtype=int)
        for val in evenly_spaced_arr:
            if not repeat:
                self._subsample_and_run(n=val)
            else:
                self._run_repeats(n=val, repeats=repeat)

    def _subsample_and_run(self, n: int) -> None:
        """helper function for subsampling data and running experiment"""
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
        """reset the best models"""
        self._best_models = {}

    def save_manifest(self) -> None:
        self.logger.log(f"Saving manifest for {self.name}")
        temp_d = {
            runner.id: runner.description
            for runner_list in self._models.values()
            for runner in runner_list
        }
        manifest_df = pd.DataFrame.from_dict(temp_d, orient="index")
        self.exporter.save_manifest_file(manifest_df)

    # todo: add logic for models where hparams are not specified, i.e. no cv needed
