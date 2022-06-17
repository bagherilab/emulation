import unittest
from unittest.mock import Mock

import pandas as pd

from permutation.runner import Runner
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams
from permutation.loader import Loader


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.model_mock = Mock(spec=Model)
        self.loader_mock = Mock(spec=Loader)

    def test_stage_initialization_with_hparam(self):
        """Check to see if correct stage is set (VAL) if no hyperparameters are passed."""
        self.model_mock.hparams = None
        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)

        self.assertEqual(testRunner.stage.name, "TRAIN")

    def test_stage_initialization_without_hparam(self):
        """Check to see if correct stage is set (TRAIN) if no hyperparameters are passed."""
        self.model_mock.hparams = Mock(spec=Hyperparams)
        testRunner = Runner(model=self.model_mock, loader=self.loader_mock)

        self.assertEqual(testRunner.stage.name, "VAL")
