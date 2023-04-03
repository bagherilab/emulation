from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class Loader(ABC):
    """
    Attributes
    ----------
    path :
        Path where data is stored
    n_total:
        Total observations
    n_working:
        Number of working observations
    n_train:
        Number of training observations
    n_test:
        Number of test observations

    Methods
    -------
    subsample(n):
        Selects n random observations to subsample,
        impacts training, testing and working data
        todo: add random state from hydra

    load_training_data():
        Return training data as dataframe with features, series with response

    load_testing_data():
        Return testing data as dataframe with features, series with response

    load_working_data():
        Return training and testing data as dataframe with features, series with response

    load_original_data():
        Return all data as dataframe with features, series with response
    """

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
        """Loading function for file interaction needs to be implemented in subclasses"""

    @abstractmethod
    def _split_data(self) -> None:
        """Method for test/train split needs to be implemented in subclasses"""

    def subsample(self, n: int, stratify: Optional[str]) -> None:
        """Sample n observations"""

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get training data

        Returns
        --------
        X
            Pandas dataframe contatining <features> variables
        y
            Pandas series containing <response> variable
        """
        return self._X.iloc[self._training_idx], self._y.iloc[self._training_idx]

    def load_testing_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get testing data

        Returns
        --------
        X
            Pandas dataframe contatining <features> variables
        y
            Pandas series containing <response> variable
        """
        return self._X.iloc[self._testing_idx], self._y.iloc[self._testing_idx]

    def load_working_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get testing and training data in one structure

        Returns
        --------
        X
            Pandas dataframe contatining <features> variables
        y
            Pandas series containing <response> variable
        """
        return self._X.iloc[self._working_idx], self._y.iloc[self._working_idx]

    def load_original_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load entire dataset

        Returns
        --------
        X
            Pandas dataframe contatining <features> variables
        y
            Pandas series containing <response> variable
        """
        return self._X, self._y

    def _set_working(self) -> None:
        """Update the working indices"""
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
    """
    Attributes
    ----------
    path :
        Path were data is stored
    n_total:
        Total observations
    n_working:
        Number of working observations
    n_train:
        Number of training observations

    n_test:
        Number of test observations

    Methods
    -------
    subsample(n):
        Selects n random observations to subsample,
        impacts training, testing and working data
        todo: add random state from hydra

    load_training_data():
        Return training data as dataframe with features, series with response

    load_testing_data():
        Return testing data as dataframe with features, series with response

    load_working_data():
        Return training and testing data as dataframe with features, series with response

    load_original_data():
        Return all data as dataframe with features, series with response
    """

    def __init__(
        self,
        path: str | Path,
        features: list[str],
        response: str,
        test_size: float = 0.3,
        stratify: Optional[str] = None,
        seed: Optional[int] = 100,
    ) -> None:
        self.path = path
        self.features = features
        self.response = response
        self.test_size = test_size
        self.seed = seed
        self.stratify = stratify
        self.stratify_labels = None
        self._load_data()
        self._split_data(stratify)

    def clean_data(self) -> None:
        """Handle missing or non-numeric data"""
        # Removed features columns with bad values
        _X_copy = self._X.copy()
        self._X = self._X.loc[:, ~(np.isnan(self._X).any(axis=0) | np.isinf(self._X)).any(axis=0)]
        removed_feature_columns = _X_copy.columns[~_X_copy.columns.isin(self._X.columns)]

        # Removed response rows with bad values
        full_data = pd.concat([self._X, self._y], axis=1)
        full_data_copy = full_data.copy()

        full_data = full_data[~full_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        removed_response_rows = full_data_copy[~full_data_copy.index.isin(full_data.index)]

        full_data.reset_index(drop=True, inplace=True)

        # Remove bad features from feature list
        self.features = [
            feature for feature in self.features if feature not in removed_feature_columns
        ]
        self._X, self._y = features_response_split(full_data, self.features, self.response)
        self._set_working()
        self._split_data(self.stratify)

        return removed_feature_columns, removed_response_rows

    def _load_data(self, index_col=0) -> None:
        """Load data from csv to _X and _y attributes"""
        data = pd.read_csv(self.path, index_col=index_col)
        data.reset_index(drop=False, inplace=True)
        self._X, self._y = features_response_split(data, self.features, self.response)
        if self.stratify:
            self.stratify_labels = data[self.stratify]
        self._set_working()

    def _split_data(self, stratify: Optional[str]) -> None:
        """Test train split implementation"""
        temp_working = pd.DataFrame(self._working_idx, columns=["working_idx"])
        if stratify:
            temp_working[stratify] = self.stratify_labels
        self._training_idx, self._testing_idx = train_test_split(
            temp_working,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=temp_working[stratify] if stratify else None,
        )
        self._training_idx = self._training_idx["working_idx"].tolist()
        self._testing_idx = self._testing_idx["working_idx"].tolist()

    def subsample(self, n: int, stratify: Optional[str] = None) -> None:
        """Sample n observations"""
        self._working_idx = self._X.sample(n).index.tolist()
        self._split_data(stratify)


def features_response_split(
    data: pd.DataFrame, features: list[str], response: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to split data into features and responses"""
    return data[features], data[response]
