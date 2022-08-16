from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split


class Loader(ABC):
    """todo"""

    path: str | Path
    features: list[str]
    response: str
    _X: pd.DataFrame
    _y: pd.Series
    _working_idx: list[int]
    _training_idx: list[int]
    _testing_idx: list[int]

    @abstractmethod
    def _load_data(self) -> None:
        """todo"""

    @abstractmethod
    def _split_data(self) -> None:
        """todo"""

    def subsample(self, n: int) -> None:
        """todo"""
        self._working_idx = self._X.sample(n).index.tolist()
        self._split_data()

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X.iloc[self._training_idx], self._y.iloc[self._training_idx]

    def load_testing_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X.iloc[self._testing_idx], self._y.iloc[self._testing_idx]

    def load_working_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X.iloc[self._working_idx], self._y.iloc[self._working_idx]

    def load_original_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """todo"""
        return self._X, self._y

    def _set_working(self) -> None:
        """todo"""
        self._working_idx = self._X.index.tolist()

    @property
    def n_total(self) -> int:
        """Property that returns the number of observations in the total data set"""
        return self._X.shape[0]

    @property
    def n_working(self) -> int:
        """Property that returns the number of observations in the working (test+train) set"""
        return len(self._working_idx)

    @property
    def n_train(self) -> int:
        """Property that returns the number of observations in the train set"""
        return len(self._training_idx)

    @property
    def n_test(self) -> int:
        """Property that returns the number of observations in the test set"""
        return len(self._testing_idx)


class CSVLoader(Loader):
    """todo"""

    def __init__(
        self,
        path: str | Path,
        features: list[str],
        response: str,
        test_size: float = 0.3,
        seed: Optional[int] = 100,
    ) -> None:
        """todo"""
        self.path = path
        self.features = features
        self.response = response
        self.test_size = test_size
        self.seed = seed
        self._load_data()
        self._split_data()

    def _load_data(self) -> None:
        """todo"""
        data = pd.read_csv(self.path)
        self._X, self._y = features_response_split(data, self.features, self.response)
        self._set_working()

    def _split_data(self) -> None:
        """todo"""
        self._training_idx, self._testing_idx = train_test_split(
            self._working_idx, test_size=self.test_size, random_state=self.seed
        )


def features_response_split(
    data: pd.DataFrame, features: list[str], response: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to split data into features and responses"""
    return data[features], data[response]
