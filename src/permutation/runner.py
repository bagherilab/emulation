from typing import Optional, Tuple

import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd

from permutation.metrics import Metric, BatchMetric, SequentialMetric
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams
from permutation.loader import Loader
from permutation.stage import Stage, IncorrectStageException


class Runner:
    """
    Runner class to manage training, alongside calls to the Model interface.
    """

    def __init__(self, model: Model, loader: Loader) -> None:
        self.model = model
        self.loader = loader

        self.training_metrics: Metric = SequentialMetric(
            f"Train RMSE, Model: {self.model.algorithm_name}"
        )
        self.test_metrics: Metric = SequentialMetric(
            f"Test RMSE, Model: {self.model.algorithm_name}"
        )
        self.cv_metrics: Optional[Metric] = None
        self.permutation_metrics: list[Metric] = []

        self.stage = Stage.TRAIN if self.model.hparams is None else Stage.VAL

    def cross_validation(self, K: int = 10) -> None:
        self.stage_check(Stage.VAL)
        X, y = self.loader.load_training_data()
        self.cv_metrics = self.model.crossval_hparams(X, y, K)

    def train(self) -> None:
        self.stage_check(Stage.TRAIN)
        X, y = self.loader.load_training_data()
        metric = self.model.fit_model(X, y)
        self.training_metrics.update(metric, len(X.index))

    def test(self) -> None:
        self.stage_check(Stage.TEST)
        X, y = self.loader.load_testing_data()
        metric = self.model.performance(X, y)
        self.test_metrics.update(metric, len(X.index))

    def permutation_testing(self) -> None:
        self.stage_check(Stage.PERM)
        X, y = self.loader.load_all_working_data()
        metrics_list = self.model.permutation(X, y)
        self.permutation_metrics.extend(metrics_list)

    def stage_check(self, correct_stage: Stage) -> None:
        if self.stage is not correct_stage:
            raise IncorrectStageException(correct_stage)
