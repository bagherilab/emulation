import unittest
from unittest.mock import Mock

from permutation.runner import Runner
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams
from permutation.loader import Loader


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.model_mock = Mock(Model)
        self.hparam_mock = Mock(Hyperparams)
        self.loader_mock = Mock(Loader)

    def test_stage_initialization_with_hparam(self):
        """Check to see if correct stage is set (VAL) if no hyperparameters are passed."""
        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)

        self.assertEqual(testRunner.stage.name, "TRAIN")

    def test_stage_initialization_without_hparam(self):
        """Check to see if correct stage is set (TRAIN) if no hyperparameters are passed."""
        testRunner = Runner(
            model=self.model_mock, loader=self.loader_mock, hparams=self.hparam_mock
        )

        self.assertEqual(testRunner.stage.name, "VAL")

    def test_loading_data_check(self):
        """Raise AttributeError if data hasn't been loaded yet for most functions"""

        def run_cv_before_loading_data():
            testRunner = Runner(
                model=self.model_mock, loader=self.loader_mock, hparams=self.hparam_mock
            )
            testRunner.cross_validation()

        def run_train_before_loading_data():
            testRunner = Runner(
                model=self.model_mock, loader=self.loader_mock, hparams=self.hparam_mock
            )
            testRunner.train()

        tests = [run_cv_before_loading_data, run_train_before_loading_data]

        for func in tests:
            with self.subTest(given=func):
                self.assertRaises(AttributeError, func)
