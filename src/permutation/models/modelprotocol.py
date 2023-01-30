from typing import Protocol, Optional

import pandas as pd
import numpy as np

from permutation.models.hyperparameters import HParams
from permutation.metrics import BatchMetric


class Model(Protocol):
    """Protocol for the Model type, requiring the following attributes"""

    algorithm_name: str
    algorithm_abv: str
    algorithm_type: str
    hparams: Optional[HParams]

    def crossval_hparams(self, X: pd.DataFrame, y: pd.Series, K: int) -> BatchMetric:
        """Performs k-fold cross validation"""
        ...

    def fit_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Fit the model to the training data"""
        ...

    def performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Check the performance of the model on test data"""
        ...

    def permutation(self, X: pd.DataFrame, y: pd.Series) -> list[BatchMetric]:
        """Permute the data"""
        ...

    def get_predicted_values(self, X: pd.DataFrame) -> np.ndarray | pd.Series:
        """Return predicted values for unlabelled data"""
        ...
