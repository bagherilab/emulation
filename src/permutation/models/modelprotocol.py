from typing import Protocol, Optional, List

import pandas as pd

from permutation.models.hyperparameters import Hyperparams
from permutation.metrics import BatchMetric


class Model(Protocol):
    """
    Protocol for the Model type, requiring the following attributes.
    """

    algorithm_name: str
    algorithm_type: str
    hparams: Optional[Hyperparams]

    def crossval_hparams(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        K: int,
    ) -> BatchMetric:
        """todo"""

    def fit_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """todo"""

    def performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """todo"""

    def permutation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[BatchMetric]:
        """todo"""
