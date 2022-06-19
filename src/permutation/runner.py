from typing import Any, Optional, Tuple

import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd

from permutation.metrics import BatchMetric, SequentialMetric
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams
from permutation.loader import Loader
from permutation.stage import Stage


class Runner:
    """
    Runner class to manage training, alongside calls to the Model interface.
    """

    def __init__(self, model: Model, loader: Loader) -> None:
        self.model = model
        self.loader = loader

        self.cv_metrics: Metric = BatchMetric(
            f"Cross Val RMSE, Model: {self.model.algorithm_name}"
        )
        self.training_metrics: Metric = SequentialMetric(
            f"Train RMSE, Model: {self.model.algorithm_name}"
        )
        self.test_metrics: Metric = SequentialMetric(
            f"Test RMSE, Model: {self.model.algorithm_name}"
        )
        self.permutation_metrics: Metric = BatchMetric(
            f"Permutation RMSE, Model: {self.model.algorithm_name}"
        )

        self.stage = Stage.TRAIN if self.model.hparams is None else Stage.VAL

    def cross_validation(self, K=10) -> None:
        X, y = self.loader.load_training_data()
        metrics = self.model.crossval_hparams(X, y, self.stage is Stage.VAL, K)
        self.cv_metrics.batchupdate(metrics)

    def train(self) -> None:
        X, y = self.loader.load_training_data()
        metrics = self.model.fit_model(X, y, self.stage is Stage.TRAIN)
        self.training_metrics.update(metrics, len(X.index))

    def test(self) -> None:
        X, y = self.loader.load_testing_data()
        metrics = self.model.performance(X, y, self.stage is Stage.TEST)
        self.test_metrics.update(metrics, len(X.index))

    def permutation_testing(self) -> None:
        X, y = self.loader.load_all_working_data()
        metrics = self.model.permutation(X, y, self.stage is Stage.PERM)
        self.permutation_metrics.update(metrics)
