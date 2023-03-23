import unittest
from unittest.mock import Mock, patch, MagicMock
import shutil
import numpy as np
import pandas as pd

from permutation.models.modelprotocol import Model
from permutation.runner import Runner
from permutation.experiments.experiment import StandardExperiment, TrainingQuantityExperiment
from permutation.models.hyperparameters import HParams
from permutation.metrics import BatchMetric
from permutation.stage import Stage, IncorrectStageException


@patch("permutation.loader.pd.read_csv")
class TestStandardExperiment(unittest.TestCase):
    def setUp(self):
        self.experiment_name = "testExperiment"
        self.export_dir = "results/test"
        self.log_dir = "logs/test"
        self.log_path = f"{self.log_dir}/{self.experiment_name}.log"
        self.data_path = "data/sample_data.csv"
        self.test_df = pd.DataFrame(
            [[0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]], columns=["x1", "x2", "y"]
        )
        self.features = ["x1", "x2"]
        self.response = "y"

        self.mock_linear_model = Mock(spec=Model)
        self.mock_linear_model.algorithm_name = "linear_test"
        self.mock_linear_model.algorithm_abv = "MLR"
        self.mock_linear_model.hparams = Mock(spec=HParams)
        self.mock_linear_model.performance = Mock(return_value=0.5)
        self.mock_linear_model.fit_model = Mock(return_value=0.5)

        self.mock_linear_model_metrics = Mock(spec=BatchMetric)
        self.mock_linear_model_metrics.stage = Stage.VAL
        self.mock_linear_model_metrics.average = 0.5
        self.mock_linear_model.crossval_hparams = Mock(return_value=self.mock_linear_model_metrics)

        self.mock_best_linear_model = Mock(spec=Model)
        self.mock_best_linear_model.algorithm_name = "best_linear_test"
        self.mock_best_linear_model.algorithm_abv = "MLR"
        self.mock_best_linear_model.hparams = Mock(spec=HParams)

        self.mock_best_linear_model_metrics = Mock(spec=BatchMetric)
        self.mock_best_linear_model_metrics.stage = Stage.VAL
        self.mock_best_linear_model_metrics.average = 0.9
        self.mock_best_linear_model.crossval_hparams = Mock(
            return_value=self.mock_best_linear_model_metrics
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.log_dir)

    def test_add_model_adds_model(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.mock_linear_model)

        self.assertEqual(experiment._n_models, 1)
        self.assertEqual(len(experiment.models), 1)

        self.assertEqual(len(experiment.algorithms), 1)
        self.assertEqual(experiment.algorithms[0], self.mock_linear_model.algorithm_abv)

    def test_add_models_adds_models(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        models = [self.mock_linear_model, self.mock_best_linear_model]
        experiment.add_models(models)

        self.assertEqual(len(experiment.models), 2)
        self.assertEqual(len(experiment.algorithms), 1)

    @patch("permutation.exporter.Exporter.save_model_json")
    def test_hyperparameter_selection_sets_best_model(self, mock_save_model, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_models([self.mock_linear_model, self.mock_best_linear_model])
        experiment.hyperparameter_selection()

        expected_model = list(experiment._models.keys())[0]
        expected_runner = str(experiment._models[expected_model][1]._UUID)
        self.assertEqual(experiment.best_models, {expected_model: expected_runner})

    @patch("permutation.exporter.Exporter.save_model_json")
    @patch("permutation.exporter.Exporter.save_predictions")
    def test_train_models_sets_runner_stage(
        self, mock_save_model, mock_save_predictions, mock_read_csv
    ):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.mock_linear_model)
        experiment.hyperparameter_selection()
        experiment.train_models()

        model_key = list(experiment._models.keys())[0]
        runner = experiment._models[model_key][0]
        expected_stage = Stage.TEST
        self.assertEqual(expected_stage, runner._stage)

    @patch("permutation.exporter.Exporter.save_model_json")
    @patch("permutation.exporter.Exporter.save_predictions")
    def test_test_models_sets_runner_stage(
        self, mock_save_model, mock_save_predictions, mock_read_csv
    ):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.mock_linear_model)
        experiment.hyperparameter_selection()
        experiment.train_models()
        experiment.test_models()

        model_key = list(experiment._models.keys())[0]
        runner = experiment._models[model_key][0]
        expected_stage = Stage.PERM
        self.assertEqual(expected_stage, runner._stage)

    @patch("permutation.exporter.Exporter.save_model_json")
    @patch("permutation.exporter.Exporter.save_predictions")
    @patch("permutation.runner.Runner.train")
    def test_train_models_calls_runner_train(
        self, mock_runner_train, mock_save_prediction, mock_save_model, mock_read_csv
    ):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.mock_linear_model)
        experiment.hyperparameter_selection()
        experiment.train_models()
        mock_runner_train.assert_called_once()

    @patch("permutation.exporter.Exporter.save_model_json")
    @patch("permutation.exporter.Exporter.save_predictions")
    @patch("permutation.runner.Runner.test")
    def test_test_models_calls_runner_test(
        self, mock_runner_test, mock_save_prediction, mock_save_model, mock_read_csv
    ):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.mock_linear_model)
        experiment.hyperparameter_selection()
        experiment.test_models()
        mock_runner_test.assert_called_once()

    @patch("permutation.exporter.Exporter.save_model_json")
    @patch("permutation.exporter.Exporter.save_predictions")
    @patch("permutation.runner.Runner.permutation_testing")
    def test_permutation_testing_calls_runner_permutation_testing(
        self, mock_runner_permutation_testing, mock_save_prediction, mock_save_model, mock_read_csv
    ):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.mock_linear_model)
        experiment.hyperparameter_selection()
        experiment.permutation_testing()
        mock_runner_permutation_testing.assert_called_once()


@patch("permutation.loader.pd.read_csv")
class TestQuantityExperiment(unittest.TestCase):
    def setUp(self):
        self.experiment_name = "testQuantityExperiment"
        self.export_dir = "results/test"
        self.log_dir = "logs/test"
        self.log_path = f"{self.log_dir}/{self.experiment_name}.log"
        self.data_path = "data/sample_data.csv"
        self.test_df = pd.DataFrame(
            [[0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]], columns=["x1", "x2", "y"]
        )
        self.features = ["x1", "x2"]
        self.response = "y"
        self.num = 5
        self.repeats = 3

    def tearDown(self) -> None:
        shutil.rmtree(self.log_dir)

    def test_subsample_and_run(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = TrainingQuantityExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
            self.num,
            self.repeats,
        )

        experiment.loader.subsample = MagicMock()

        experiment._run_standard_experiment = MagicMock()
        experiment._subsample_and_run(experiment._num)
        experiment.loader.subsample.assert_called_with(self.num)
        experiment._run_standard_experiment.assert_called_once()

    def test_run_repeats(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = TrainingQuantityExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
            self.num,
            self.repeats,
        )

        experiment.loader.subsample = MagicMock()

        experiment._run_standard_experiment = MagicMock()
        experiment._run_repeats(experiment._num, experiment._repeat)
        self.assertEqual(experiment.loader.subsample.call_count, self.repeats)
        experiment.loader.subsample.assert_called_with(self.num)
