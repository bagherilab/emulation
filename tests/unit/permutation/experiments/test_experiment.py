import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from permutation.models.modelprotocol import Model
from permutation.runner import Runner
from permutation.loader import Loader
from permutation.experiments.experiment import StandardExperiment
from permutation.models.hyperparameters import HParams
from permutation.metrics import BatchMetric
from permutation.stage import Stage

@patch("permutation.loader.pd.read_csv")
class TestStandardExperiment(unittest.TestCase):
    def setUp(self):
        self.experiment_name = "TestExperiment"
        self.export_dir = "results/test"
        self.log_dir = "logs/test"
        self.data_path = "data/sample_data.csv"
        self.test_df = pd.DataFrame([[0, 0, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]], columns=["x1", "x2", "y"])
        self.features = ["x1", "x2"]
        self.response = "y"

        self.linear_model_mock = Mock(spec=Model)
        self.linear_model_mock.algorithm_name = "linear_test"
        self.linear_model_mock.algorithm_abv = "MLR"
        self.linear_model_mock.hparams = Mock(spec=HParams)

        self.best_linear_model_mock = Mock(spec=Model)
        self.best_linear_model_mock.algorithm_name = "best_linear_test"
        self.best_linear_model_mock.algorithm_abv = "MLR"
        self.best_linear_model_mock.hparams = Mock(spec=HParams)

    def test_add_model(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df
        
        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.linear_model_mock)
        
        self.assertEqual(experiment._n_models, 1)
        self.assertEqual(len(experiment.models), 1)

        self.assertEqual(len(experiment.algorithms), 1)
        self.assertEqual(experiment.algorithms[0], self.linear_model_mock.algorithm_abv)

    def test_add_models(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df
        
        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        models = [self.linear_model_mock, self.best_linear_model_mock]
        experiment.add_models(models)
        
        self.assertEqual(len(experiment.models), 2)
        self.assertEqual(len(experiment.algorithms), 1)

    @patch("permutation.exporter.Exporter.save_model_json")
    def test_hyperparameter_selection(self, mock_save_model, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        self.linear_model_mock_metrics = Mock(spec=BatchMetric)
        self.linear_model_mock_metrics.stage = Stage.VAL
        self.linear_model_mock_metrics.average = 0.5
        self.linear_model_mock.crossval_hparams = Mock(return_value = self.linear_model_mock_metrics)

        self.best_linear_model_mock_metrics = Mock(spec=BatchMetric)
        self.best_linear_model_mock_metrics.stage = Stage.VAL
        self.best_linear_model_mock_metrics.average = 0.9
        self.best_linear_model_mock.crossval_hparams = Mock(return_value = self.best_linear_model_mock_metrics)

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_models([self.linear_model_mock, self.best_linear_model_mock])
        experiment.hyperparameter_selection()

        expected_model = list(experiment._models.keys())[0]
        expected_runner = str(experiment._models[expected_model][1]._UUID)
        self.assertEqual(experiment.best_models, {expected_model: expected_runner})


    def test_train_models(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.linear_model_mock)
        experiment.train_models()
        
        model_key = list(experiment._models.keys())[0]
        runner = experiment._models[model_key][0]


    def test_test_models(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        experiment = StandardExperiment(
            self.experiment_name,
            self.export_dir,
            self.log_dir,
            self.data_path,
            self.features,
            self.response,
        )

        experiment.add_model(self.linear_model_mock)
        experiment.train_models()
        experiment.test_models()

        model_key = list(experiment._models.keys())[0]
        runner = experiment._models[model_key][0]

        
    

