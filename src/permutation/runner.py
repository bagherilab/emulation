from typing import Any, Optional, Tuple

import sklearn as sk
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

    def __init__(
        self,
        model: Model,
        loader: Loader,
        hparams: Optional[Hyperparams] = None,
        subsample_n: Optional[int] = None,
    ) -> None:
        self.model = model
        self.loader = loader

        self.training_metrics: SequentialMetric = None
        self.test_metrics: BatchMetric = None
        self.permutation_metrics: BatchMetric = None

        self.X: pd.DataFrame = None
        self.y: pd.Series = None

        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None

        self.stage = Stage.TRAIN if hparams is None else Stage.VAL
        self.hparams = hparams

    def load_data(self, features: list[str], response: str, test_size=0.3, seed=100):
        temp_df = self.loader.load()

        if subsample_n:
            temp_df = self.subsample_data(temp_df, self.subsample_n)

        self.X, self.y = features_response_split(temp_df, features, response)

        self.X_train, self.X_test, self.y_train, self.y_test = split(
            self.X, self.y, test_size, seed
        )

    def unload_data(self) -> None:
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def cross_validation(self) -> None:
        self._check_data()
        metrics = self.model.crossval_hparams(
            self.X_train, self.y_train, self.hparams, self.stage is Stage.VAL
        )
        self.hparams.update_cv_metrics(metrics)

    def train(self) -> None:
        self._check_data()
        metrics = self.model.fit_model(
            self.X_train, self.y_train, self.stage is Stage.TRAIN
        )
        self.training_metrics.update(metrics, len(self.X_train.index))

    def test(self) -> None:
        metrics = self.model.performance(
            self.X_test, self.y_test, self.stage is Stage.TEST
        )
        self.test_metrics.update(metrics, len(self.X_train.index))

    def permutation_testing(self) -> None:
        metrics = self.model.permutation(self.X, self.y, self.stage is Stage.PERM)
        self.permutation_metrics.update(metrics)

    def _check_data(self) -> None:
        if not self.x:
            raise AttributeError("Data has not been loaded yet.")


def features_response(
    data: pd.DataFrame, features: list[str], response: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to split data into features and responses"""
    return data[features], data[response]


def training_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """helper function for training and test splits"""
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test
