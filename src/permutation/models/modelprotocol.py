from typing import Protocol, Any
from dataclasses import dataclass, field

from permutation.stage import Stage
from permutation.models.hyperparameters import Hyperparams


class Model(Protocol):
    """
    Protocol for the Model type, requiring the following attributes.
    """

    algorithm_name: str
    algorithm_type: str
    hparams: Hyperparams

    def crossval_hparams(self, x: Any, y: Any, stage_check: bool) -> list[float]:
        ...

    def fit_model(self, x: Any, y: Any, stage_check: bool) -> float:
        ...

    def performance(self, x: Any, y: Any, stage_check: bool) -> float:
        ...

    def permutation(self, x: Any, y: Any, stage_check: bool) -> list[float]:
        ...
