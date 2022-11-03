from typing import Protocol, Optional

import pandas as pd
import numpy as np

from permutation.models.hyperparameters import HParams
from permutation.metrics import BatchMetric


class Model(Protocol):
    """
    Protocol for the Model type, requiring the following attributes.
    """

    algorithm_name: str
    algorithm_abv: str
    algorithm_type: str
    hparams: Optional[HParams]

    def crossval_hparams(self, X: pd.DataFrame, y: pd.Series, K: int) -> BatchMetric:
        """todo"""

    def fit_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """todo"""

    def performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """todo"""

    def permutation(self, X: pd.DataFrame, y: pd.Series) -> list[BatchMetric]:
        """todo"""

    def get_predicted_values(self, X: pd.DataFrame) -> np.ndarray | pd.Series:
        """todo"""
