from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd

import sklearn.model_selection


class Loader(ABC):
    """todo"""

    path: str | Path
    features: list[str]
    response: str
    _X: Optional[pd.DataFrame]
    _y: Optional[pd.Series]
    _X_working: Optional[pd.DataFrame]
    _y_working: Optional[pd.Series]
    _X_training: Optional[pd.DataFrame]
    _y_training: Optional[pd.Series]
    _X_testing: Optional[pd.DataFrame]
    _y_testing: Optional[pd.Series]

    @abstractmethod
    def _load_data(self) -> None:
        """todo"""
        ...

    @abstractmethod
    def _split_data(self):
        """todo"""
        ...

    def subsample(self, n) -> None:
        """todo"""
        self._X_working = self._X.sample(n)
        self._y_working = self._y_working[self._X_working.index]
        self._split_data()

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X_training, self._y_training

    def load_testing_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X_testing, self._y_testing

    def load_all_working_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X_working, self._y_working

    def load_original_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X, self._y

    def _set_working(self) -> None:
        """todo"""
        self._X_working, self._y_working = self._X.copy(deep=True), self._y.copy(deep=True)

    def unload_data(self) -> None:
        """todo"""
        self._X = None
        self._y = None
        self._X_working = None
        self._y_working = None
        self._X_training = None
        self._y_training = None
        self._X_testing = None
        self._y_testing = None


class CSVLoader(Loader):
    """todo"""

    def __init__(
        self,
        path: str,
        features: list[str],
        response: str,
        test_size: float = 0.3,
        seed: Optional[int] = 100,
    ):
        """todo"""
        self.path: str = path
        self.features = features
        self.response = response
        self.test_size = test_size
        self.seed = seed
        self._X = None
        self._y = None
        self._X_working = None
        self._y_working = None
        self._X_training = None
        self._y_training = None
        self._X_testing = None
        self._y_testing = None

        self._load_data()
        self._split_data()

    def _load_data(self):
        """todo"""
        data = pd.read_csv(self.path)
        self._X, self._y = features_response_split(data, self.features, self.response)
        self._set_working()

    def _split_data(self):
        """todo"""
        (
            self._X_training,
            self._X_testing,
            self._y_training,
            self._y_testing,
        ) = stage_split(self._X_working, self._y_working, self.test_size, self.seed)


def features_response_split(
    data: pd.DataFrame, features: list[str], response: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to split data into features and responses"""
    return data[features], data[response]


def stage_split(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """helper function for training and test splits"""

    test_train_tuple = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_training, X_testing, y_training, y_testing = test_train_tuple

    return X_training, X_testing, y_training, y_testing
