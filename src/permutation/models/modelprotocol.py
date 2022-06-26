from typing import Protocol
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from permutation.stage import Stage
from permutation.models.hyperparameters import Hyperparams


class Model(Protocol):
    """
    Protocol for the Model type, requiring the following attributes.
    """

    algorithm_name: str
    algorithm_type: str
    hparams: Hyperparams

    def crossval_hparams(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        hparams: Hyperparams,
        stage_check: bool,
    ) -> list[float]:
        ...

    def fit_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        stage_check: bool,
    ) -> float:
        ...

    def performance(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        stage_check: bool,
    ) -> float:
        ...

    def permutation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        stage_check: bool,
    ) -> list[float]:
        ...
