import unittest
from unittest.mock import Mock, MagicMock

import pandas as pd

from permutation.runner import Runner
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams
from permutation.loader import Loader
from permutation.stage import Stage


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

        self.assertEqual(testRunner.stage.name, "TRAIN")

    def test_stage_initialization_without_hparam(self) -> None:
        """Check to see if correct stage is set (TRAIN) if no hyperparameters are passed."""
        self.model_mock.hparams = Mock(spec=Hyperparams)
        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)

        self.assertEqual(testRunner.stage.name, "VAL")

    def test_cross_validation_hparams_called(self) -> None:
        self.model_mock.crossval_hparams = Mock(return_value=[1.0] * 10)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.cross_validation()

        self.model_mock.crossval_hparams.assert_called_once_with(
            *self.load_data_return, True, 10
        )

    def test_cross_validation_hparams_called_at_incorrect_stage(self) -> None:
        self.model_mock.crossval_hparams = Mock(return_value=[1.0] * 10)
        self.model_mock.hparams = None
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.cross_validation()

        self.model_mock.crossval_hparams.assert_called_once_with(
            *self.load_data_return, False, 10
        )

    def test_fit_model_called(self) -> None:
        self.model_mock.fit_model = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.stage = Stage.TRAIN
        testRunner.train()

        self.model_mock.fit_model.assert_called_once_with(*self.load_data_return, True)

    def test_fit_model_called_at_incorrect_stage(self) -> None:
        self.model_mock.fit_model = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_training_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.stage = Stage.TEST
        testRunner.train()

        self.model_mock.fit_model.assert_called_once_with(*self.load_data_return, False)

    def test_performance_called(self) -> None:
        self.model_mock.performance = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_testing_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.stage = Stage.TEST
        testRunner.test()

        self.model_mock.performance.assert_called_once_with(
            *self.load_data_return, True
        )

    def test_performance_called_at_incorrect_stage(self) -> None:
        self.model_mock.performance = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_testing_data = Mock(return_value=self.load_data_return)

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.stage = Stage.TRAIN
        testRunner.test()

        self.model_mock.performance.assert_called_once_with(
            *self.load_data_return, False
        )

    def test_permutation_called(self) -> None:
        self.model_mock.permutation = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_all_working_data = Mock(
            return_value=self.load_data_return
        )

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.stage = Stage.PERM
        testRunner.permutation_testing()

        self.model_mock.permutation.assert_called_once_with(
            *self.load_data_return, True
        )

    def test_permutation_called_at_incorrect_stage(self) -> None:
        self.model_mock.permutation = Mock(return_value=1.0)
        self.model_mock.hparams = Mock(spec=Hyperparams)
        self.loader_mock.load_all_working_data = Mock(
            return_value=self.load_data_return
        )

        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)
        testRunner.stage = Stage.TEST
        testRunner.permutation_testing()

        self.model_mock.permutation.assert_called_once_with(
            *self.load_data_return, False
        )
