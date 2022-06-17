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
    Runner class to manage training, test data, alongside calls to the Model interface.
    Runner contains the data from multiple runs of the same model with the same hyperparameters.
    """

    def __init__(self, model: Model, loader: Loader) -> None:
        self.model = model
        self.loader = loader

        self.cv_metrics: BatchMetric = None
        self.training_metrics: SequentialMetric = None
        self.test_metrics: SequentialMetric = None
        self.permutation_metrics: BatchMetric = None

        self.stage = Stage.TRAIN if self.model.hparams is None else Stage.VAL

    def cross_validation(self, K=10) -> None:
        metrics = self.model.crossval_hparams(
            self.loader.X_train,
            self.loader.y_train,
            self.stage is Stage.VAL,
            K,
        )
        self.cv_metrics.batch_update(metrics)

    def train(self) -> None:
        metrics = self.model.fit_model(
            self.loader.X_train, self.loader.y_train, self.stage is Stage.TRAIN
        )
        self.training_metrics.update(metrics, len(self.X_train.index))

    def test(self) -> None:
        metrics = self.model.performance(
            self.loader.X_test, self.loader.y_test, self.stage is Stage.TEST
        )
        self.test_metrics.update(metrics, len(self.X_train.index))

    def permutation_testing(self) -> None:
        metrics = self.model.permutation(
            self.loader.X, self.loader.y, self.stage is Stage.PERM
        )
        self.permutation_metrics.update(metrics)
