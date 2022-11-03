import unittest
from unittest.mock import Mock, MagicMock

import pandas as pd

from permutation.runner import Runner
from permutation.loader import Loader
from permutation.stage import Stage, IncorrectStageException
from permutation.metrics import BatchMetric
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import HParams


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.model_mock = Mock(spec=Model)
        self.model_mock.algorithm_name = Mock(return_value="test")

        self.loader_mock = Mock(spec=Loader)
        df_mock = MagicMock(spec=pd.DataFrame)
        series_mock = MagicMock(spec=pd.Series)
        self.load_data_return = (df_mock, series_mock)

    def test_stage_initialization_with_hparam(self) -> None:
        """Check to see if correct stage is set (VAL) if no hyperparameters are passed."""
        self.model_mock.hparams = None
        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)

        self.assertEqual(testRunner._stage.name, "TRAIN")

    def test_stage_initialization_without_hparam(self) -> None:
        """Check to see if correct stage is set (TRAIN) if no hyperparameters are passed."""
        self.model_mock.hparams = Mock(spec=HParams)
        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)

        self.assertEqual(testRunner._stage.name, "VAL")

    def test_cross_validation_hparams_called(self) -> None:
        self.model_mock.crossval_hparams = Mock(return_value=[1.0] * 10)
        self.model_mock.hparams = Mock(spec=HParams)
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.cross_validation()

        self.model_mock.crossval_hparams.assert_called_once_with(*self.load_data_return, 10)

    def test_cross_validation_hparams_called_at_incorrect_stage(self) -> None:
        self.model_mock.crossval_hparams = Mock(return_value=[1.0] * 10)
        self.model_mock.hparams = None
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        def call_val_without_hparams():
            testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
            testRunner.cross_validation()

        self.assertRaises(IncorrectStageException, call_val_without_hparams)

    def test_fit_model_called(self) -> None:
        self.model_mock.fit_model = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=HParams)
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner._stage = Stage.TRAIN
        testRunner.train()

        self.model_mock.fit_model.assert_called_once_with(*self.load_data_return)

    def test_fit_model_called_at_incorrect_stage(self) -> None:
        self.model_mock.fit_model = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=HParams)
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        def call_train_at_wrong_stage():
            testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
            testRunner._stage = Stage.TEST
            testRunner.train()

        self.assertRaises(IncorrectStageException, call_train_at_wrong_stage)

    def test_performance_called(self) -> None:
        self.model_mock.performance = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=HParams)
        self.loader_mock.load_testing_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner._stage = Stage.TEST
        testRunner.test()

        self.model_mock.performance.assert_called_once_with(*self.load_data_return)

    def test_performance_called_at_incorrect_stage(self) -> None:
        self.model_mock.performance = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=HParams)
        self.loader_mock.load_testing_data = Mock(return_value=self.load_data_return)

        def call_test_at_wrong_stage():
            testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
            testRunner._stage = Stage.TRAIN
            testRunner.test()

        self.assertRaises(IncorrectStageException, call_test_at_wrong_stage)

    def test_permutation_called(self) -> None:
        metric_mock = MagicMock(spec=BatchMetric)
        list_of_metric_mocks = [metric_mock] * 10
        self.model_mock.permutation = Mock(return_value=list_of_metric_mocks)
        self.model_mock.hparams = Mock(spec=HParams)
        self.loader_mock.load_working_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner._stage = Stage.PERM
        testRunner.permutation_testing()

        self.model_mock.permutation.assert_called_once_with(*self.load_data_return)
        self.assertEqual(testRunner.permutation_metrics, list_of_metric_mocks)

    def test_permutation_called_at_incorrect_stage(self) -> None:
        self.model_mock.hparams = Mock(spec=HParams)

        def call_perm_at_wrong_stage():
            testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
            testRunner._stage = Stage.TEST
            testRunner.permutation_testing()

        self.assertRaises(IncorrectStageException, call_perm_at_wrong_stage)
