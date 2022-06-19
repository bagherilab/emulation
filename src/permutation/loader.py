from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd

import sklearn.model_selection


class Loader(ABC):
    path: str | Path
    features: list[str]
    response: str
    _X: pd.DataFrame
    _y: pd.Series
    _X_train: pd.DataFrame
    _X_test: pd.DataFrame
    _y_train: pd.Series
    _y_test: pd.Series

    @abstractmethod
    def _load_data(self, subsample_n=None) -> None:
        ...

    @abstractmethod
    def _split_data(self):
        ...

    def subsample(self, n) -> None:
        self._X_working = self._X.sample(n)
        self._y_working = self._y_working[self._X_working.index]
        self._split_data()

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self._X_train, self._y_train

    def load_testing_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self._X_test, self._y_test

    def load_all_working_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self._X_working, self._y_working

    def load_original_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self._X, self._y

    def _set_working(self) -> None:
        self._X_working, self._y_working = self._X.copy(deep=True), self._y.copy(
            deep=True
        )

    def unload_data(self) -> None:
        self._X = None
        self._y = None
        self._X_working = None
        self._y_working = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None


class CSVLoader(Loader):
    def __init__(
        self,
        path: str,
        features: list[str],
        response: str,
        test_size: float = 0.3,
        seed: Optional[int] = 100,
    ):
        self.path: str = path

        self.features = features
        self.response = response
        self.test_size = test_size
        self.seed = seed

        self._X: pd.DataFrame = None
        self._y: pd.Series = None

        self._X_working: pd.DataFrame = None
        self._y_working: pd.Series = None

        self._X_train: pd.DataFrame = None
        self._X_test: pd.DataFrame = None
        self._y_train: pd.Series = None
        self._y_test: pd.Series = None

        self._load_data()
        self._split_data()

    def _load_data(self):
        data = pd.read_csv(self.path)
        self._X, self._y = features_response_split(data, self.features, self.response)
        self._set_working()

    def _split_data(self):
        self._X_train, self._X_test, self._y_train, self._y_test = stage_split(
            self._X_working, self._y_working, self.test_size, self.seed
        )


def features_response_split(
    data: pd.DataFrame, features: list[str], response: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to split data into features and responses"""
    return data[features], data[response]


def stage_split(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """helper function for training and test splits"""
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test
