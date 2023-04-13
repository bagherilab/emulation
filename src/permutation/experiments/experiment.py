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
    """
    ABC for experiment types

    Attributes
    ----------
    name :
        Location where files will be stored
    exporter :
        Exporter object where the experiment will export results
    logger :
        Logger object where the experiment will log status
    loader :
        Loader object where the experiment will load data
    models:
        Dictionary of models to test
    algorithms:
        List of algorithms to test
    best_models:
        Dictionary of best models for each algorithm

    Methods
    -------
    run():
        Run the experiment

    hyperparameter_selection():
        Perform hyperparameter selection on all models for all algorithms
        Add models to best_models

    train_models():
        Train models in best_models

    test_models():
        Train_models() must be run first.
        Get performance of models in best_models

    add_model(model: Model):
        Add a model to the experiment

    add_models(models: list[Model]):
        Add a list of models to the experiment

    permutation_testing():
        Train_models() must be run first.
        Get permutation testing results of all models in _best_models

    save_manifest():
        Save a manifest of all models in the experiment

    save_train_test():
        Save the training and testing data as they were used for the experiment


    """

    name: str
    exporter: Exporter
    logger: Logger
    loader: Loader
    _models: dict[str, list[Runner]]
    _n_models: int
    _model_ids: set = set()

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

    def hyperparameter_selection(self) -> None:
        """Perform hparam selection using cv"""
        self._get_best_cv_model()

    def train_models(self) -> None:
        """Train models in best_model"""
        for algorithm, runner in self._best_models.items():
            runner.train()
            runner.set_stage(Stage.TEST)
            self._log_training_performance(algorithm)

    def test_models(self) -> None:
        """Test trained best_model performances"""
        for algorithm, runner in self._best_models.items():
            runner.test()
            runner.set_stage(Stage.PERM)
            self._log_test_performance(algorithm)

    def _clean_data(self, clean_data_flag: bool) -> None:
        if clean_data_flag:
            removed_feature_columns, removed_response_rows = self.loader.clean_data()

            self.logger.log(
                f"Removed the following features from the dataset due to missing, infinity, or nan values in the column:"
            )
            for feature in removed_feature_columns:
                self.logger.log(f"{feature}")

            self.logger.log(
                f"Removed {len(removed_response_rows)} row(s) from data due to missing, infinity, or nan values in the response column"
            )

        else:
            self.logger.log(
                "Data not cleaned. Program may crash if missing or nan values are present"
            )

    def add_model(self, model: Model) -> None:
        """Add a model to test to the experiment"""
        self._check_algorithm_in_models(model.algorithm_abv)
        runner = Runner(model, self.loader)
        self._models[model.algorithm_abv].append(runner)
        model_id = self._check_ids(runner)
        self.logger.log(f"Added {model_id}.")
        self._n_models += 1

    def add_models(self, models: list[Model]) -> None:
        """Add a list of models to the experiment"""
        for model in models:
            self.add_model(model)

    def permutation_testing(self) -> None:
        """Perform permutation testing"""
        for algorithm, runner in self._best_models.items():
            runner.permutation_testing()
            self._log_perm_performance(algorithm)

    def save_manifest(self) -> None:
        """Save manifest as a CSV"""
        self.logger.log(f"Saving manifest for {self.name}")
        temp_d = {
            runner.id: runner.description
            for runner_list in self._models.values()
            for runner in runner_list
        }
        manifest_df = pd.DataFrame.from_dict(temp_d, orient="index")
        self.exporter.save_manifest_file(manifest_df)

    def save_train_test(self) -> None:
        """Save the training and testing data"""
        self.logger.log(f"Saving train/test data for {self.name}")
        train_df, test_df = self.loader.load_working_data()
        self.exporter.save_train_test(train_df, test_df)

    def _get_best_cv_model(self) -> None:
        """Run CV for all models and identify best model"""
        progress_counter = 0

        for algorithm, runner_list in self._models.items():
            for runner in runner_list:
                runner.cross_validation()
                progress_counter += 1
                self.logger.log(f"Progress: {progress_counter} of {self._n_models} - {runner.id}")

            cv_list = [runner.cv_metrics.average for runner in runner_list]  # type: ignore
            best_value = max(cv_list)
            model_index = next(i for i, val in enumerate(cv_list) if val == best_value)
            self._log_hyperparameter_selection(algorithm)
            self._update_best_model(algorithm, model_index)

    def _reset_best_models(self) -> None:
        """Reset the best models"""
        self._best_models: dict = {}

    def _check_algorithm_in_models(self, name: str) -> None:
        """Check that an algorithm exists in a model"""
        if name not in self._models:
            self._models[name] = []

    def _update_best_model(self, algorithm: str, model_index: int) -> None:
        """Update best_model"""
        self._best_models[algorithm] = self._models[algorithm][model_index]
        self._best_models[algorithm].set_stage(Stage.TRAIN)
        self.exporter.save_model_json(self._best_models[algorithm])

    def _check_ids(self, runner: Runner) -> str:
        """Reset the runner's ID if it exists in already"""
        while runner.id in self._model_ids:
            self.logger.log(f"Reset id for {runner.description}")
            runner.reset_id()

        return runner.id

    def _log_training_performance(self, algorithm: str) -> None:
        """Log the training performance"""
        runner = self._best_models[algorithm]
        self.exporter.metric_to_csv(algorithm, runner.id, runner.training_metrics)
        self.exporter.save_predictions(algorithm, runner)

    def _log_test_performance(self, algorithm: str) -> None:
        """Log test performance"""
        runner = self._best_models[algorithm]
        self.exporter.metric_to_csv(algorithm, runner.id, runner.testing_metrics)
        self.exporter.save_predictions(algorithm, runner)

    def _log_hyperparameter_selection(self, algorithm: str) -> None:
        """Log the data from cross validation"""
        self.logger.log(f"Trying to log CV for {algorithm}")
        for runner in self._models[algorithm]:
            # if None, an IncorrectStageException will be thrown before the next line runs
            self.exporter.metric_to_csv(algorithm, runner.id, runner.cv_metrics)  # type: ignore
            self.logger.log(f"Logging CV for {runner.id}")

    def _log_perm_performance(self, algorithm: str) -> None:
        """Log the permutation performance"""
        runner = self._best_models[algorithm]
        for perm_metric in runner.permutation_metrics:
            self.exporter.metric_to_csv(algorithm, f"{runner.id}_{perm_metric.name}", perm_metric)


class StandardExperiment(Experiment):
    """
    Standard experiment class


    Arguments
    ---------
    experiment_name: str
        Name of the experiment
    export_dir: str
        Directory to export the experiment data
    log_dir: str
        Directory to log the experiment data
    data_path: str
        Path to the data
    features: list[str]
        List of features to use
    response: str
        Response variable

    Attributes
    ----------
    name :
        Location where files will be stored
    exporter :
        Exporter object where the experiment will export results
    logger :
        Logger object where the experiment will log status
    loader :
        Loader object where the experiment will load data
    models:
        Models that have been added to the experiment
    algorithms :
        Algorithms that have been added to the experiment
    best_models :
        Hyperparameter models that have been identified as the best model through CV

    Methods
    -------
    run()
        run a standard experiment
    """

    def __init__(
        self,
        experiment_name: str,
        export_dir: str,
        log_dir: str,
        data_path: str,
        features: list[str],
        response: str,
        stratify: Optional[str] = None,
        clean_data_flag: bool = False,
    ) -> None:
        self.name = experiment_name
        self.exporter: Exporter = Exporter(self.name, export_dir)
        self.logger: Logger = ExperimentLogger(self.name, log_dir)
        self.loader: Loader = CSVLoader(data_path, features, response, stratify=stratify)
        self._features = features
        self._response = response
        self._models: dict[str, list[Runner]] = {}
        self._model_ids: set = set()
        self._n_models = 0
        self._best_models: dict[str, Runner] = {}
        self.stratify = stratify
        self._log_initialization()
        self._clean_data(clean_data_flag)

    def run(self) -> None:
        self._run_experiment()

    def _run_experiment(self) -> None:
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

    def _log_initialization(self) -> None:
        self.logger.log(f"(Standard Experiment) {self.name} initialized.")

    def _clean_data(self, clean_data_flag: bool) -> None:
        if clean_data_flag:
            (
                removed_feature_columns,
                removed_response_rows,
                mult_comp_rows,
            ) = self.loader.clean_data()

            if len(mult_comp_rows) > 0:
                self.logger.log(
                    f"Removed {len(mult_comp_rows)} row(s) from data due to multiple components in the network"
                )

            if len(removed_feature_columns) > 0:
                self.logger.log(
                    f"Removed the following features from the dataset due to missing, infinity, or nan values in the column:"
                )
                for feature in removed_feature_columns:
                    self.logger.log(f"{feature}")

            if len(removed_response_rows) > 0:
                self.logger.log(
                    f"Removed {len(removed_response_rows)} row(s) from data due to missing, infinity, or nan values in the response column"
                )

        else:
            self.logger.log(
                "Data not cleaned. Program may crash if missing or nan values are present"
            )

    @property
    def models(self) -> list[str]:
        """Returns a list of model IDs for the experiment"""
        list_of_models = []
        for _, runner_list in self._models.items():
            temp_name_list = [runner.id for runner in runner_list]
            list_of_models.extend(temp_name_list)
        return list_of_models

    @property
    def algorithms(self) -> list[str]:
        """Returns algorithms included in the experiment"""
        return list(self._models.keys())

    @property
    def best_models(self) -> dict[str, str]:
        """Returns dictionary and name of best models"""
        return {alg: runner.id for alg, runner in self._best_models.items()}

    # todo: add logic for models where hparams are not specified, i.e. no cv needed


class TrainingQuantityExperiment(Experiment):
    """
    A class to set up and initialize an experiment to test how much training data is required

    Arguments
    ----------
    experiment_name :
        Name of experiment for file management purposes
    log_dir :
        Directory to save experiment results
    data_path :
        Location of original datafile
    features :
        List of features in the dataset to use to train models
    response :
        Response variable to use as the target variable for training
    num :
        Number of training data points to use for each experiment
    repeat :
        Number of times to repeat the experiment

    Attributes
    ----------
    name :
        Location where files will be stored
    exporter :
        Exporter object where the experiment will export results
    logger :
        Logger object where the experiment will log status
    loader :
        Loader object where the experiment will load data
    models:
        Models that have been added to the experiment
    algorithms :
        Algorithms that have been added to the experiment
    best_models :
        Hyperparameter models that have been identified as the best model through CV

    Methods
    -------
    run()
        run a training quantity experiment
    """

    def __init__(
        self,
        experiment_name: str,
        export_dir: str,
        log_dir: str,
        data_path: str,
        features: list[str],
        response: str,
        num: int = 10,
        repeat: Optional[int] = None,
        stratify: Optional[str] = None,
        clean_data_flag: bool = False,
    ) -> None:
        self.name = experiment_name
        self.exporter: Exporter = Exporter(self.name, export_dir)
        self.logger: Logger = ExperimentLogger(self.name, log_dir)
        self.loader: Loader = CSVLoader(data_path, features, response, stratify=stratify)
        self._features = features
        self._response = response
        self._models: dict[str, list[Runner]] = {}
        self._model_ids: set = set()
        self._n_models = 0
        self._best_models: dict[str, Runner] = {}
        self._num = num
        self._repeat = repeat
        self.stratify = stratify
        self._log_initialization()
        self._clean_data(clean_data_flag)

    def run(self) -> None:
        self._run_training_quantity_experiment(self._num, self._repeat)

    def _log_initialization(self) -> None:
        self.logger.log(f"(Training Quantity Experiment) {self.name} initialized.")

    def _clean_data(self, clean_data_flag: bool) -> None:
        if clean_data_flag:
            (
                removed_feature_columns,
                removed_response_rows,
                mult_comp_rows,
            ) = self.loader.clean_data()

            if len(mult_comp_rows) > 0:
                self.logger.log(
                    f"Removed {len(mult_comp_rows)} row(s) from data due to multiple components in the network"
                )

            self.logger.log(
                f"Removed the following features from the dataset due to missing, infinity, or nan values in the column:"
            )
            for feature in removed_feature_columns:
                self.logger.log(f"{feature}")

            self.logger.log(
                f"Removed {len(removed_response_rows)} row(s) from data due to missing, infinity, or nan values in the response column"
            )

        else:
            self.logger.log(
                "Data not cleaned. Program may crash if missing or nan values are present"
            )

    def _run_training_quantity_experiment(self, num: int, repeat: Optional[int]) -> None:
        """
        Run the member methods associated with the experiment.
        Repeats standard experiment times
        """
        self._reset_best_models()
        training_size_arr = np.linspace(30, self.loader.n_total, num=num, dtype=int)
        largest_to_smallest = np.flip(training_size_arr)
        for val in largest_to_smallest:
            if not repeat:
                self._subsample_and_run(n=val)
            else:
                self._run_repeats(n=val, repeats=repeat)

            # Reset runner stage to train for next data subset
            for _, runner in self._best_models.items():
                runner.set_stage(Stage.TRAIN)

    def _subsample_and_run(self, n: int) -> None:
        """Helper function for subsampling data and running experiment"""
        self.loader.subsample(n, self.stratify)
        self._run_experiment()

    def _run_repeats(self, n: int, repeats: int) -> None:
        """
        Helper function for training quantity experiment,
        handles naming and rerunning experiment
        """
        original = self.name
        for i in range(repeats):
            temp_name = f"{original}_{i}"
            self.name = temp_name
            self._subsample_and_run(n=n)
        self.name = original

    def _run_experiment(self) -> None:
        """
        Run the member methods associated with the experiment.
        1) Hyperparameter selection via cross validation on the largest data subset.
        2) Model training of the best models selected in (1) for all data subsets
        3) Model performance testing of the best models selected in (1) for all data subsets
        4) Permutation testing of the best models selected in (1) for all data subsets
        """
        try:
            self.hyperparameter_selection()
        except IncorrectStageException:
            # No hyperparameters were passed TODO: add logger functionality to log to file
            pass
        self.train_models()
        self.test_models()
        self.permutation_testing()

    @property
    def models(self) -> list[str]:
        """Returns a list of model IDs for the experiment"""
        list_of_models = []
        for _, runner_list in self._models.items():
            temp_name_list = [runner.id for runner in runner_list]
            list_of_models.extend(temp_name_list)
        return list_of_models

    @property
    def algorithms(self) -> list[str]:
        """Returns algorithms included in the experiment"""
        return list(self._models.keys())

    @property
    def best_models(self) -> dict[str, str]:
        """Returns dictionary and name of best models"""
        return {alg: runner.id for alg, runner in self._best_models.items()}
