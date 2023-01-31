import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from permutation.models.mlr import MLR
from permutation.models.rf import RF
from permutation.runner import Runner
from permutation.loader import Loader
from permutation.experiments.experiment import StandardExperiment

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

        self.linear_model = MLR()
        self.rf_model = RF()

        self.runner_mock = Mock(spec=Runner)
        self.loader_mock = Mock(spec=Loader)
        self.runner_mock.model = self.linear_model
        self.loader_mock.load_training_data.return_value = self.test_df[["x1", "x2"]], self.test_df["y"]

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

        experiment.add_model(self.linear_model)
        
        self.assertEqual(experiment._n_models, 1)
        self.assertEqual(len(experiment.models), 1)

        self.assertEqual(len(experiment.algorithms), 1)
        self.assertEqual(experiment.algorithms[0], self.linear_model.algorithm_abv)

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

        models = [self.linear_model, self.rf_model]
        experiment.add_models(models)
        
        self.assertEqual(len(experiment.models), 2)
        self.assertEqual(len(experiment.algorithms), 2)

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

        experiment.add_model(self.linear_model)
        experiment.train_models()
        print(self.runner_mock.testing_metrics)
        # experiment.test_models()

