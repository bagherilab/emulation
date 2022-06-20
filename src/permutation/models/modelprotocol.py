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

    def crossval_hparams(
        self, X: Any, y: Any, hparams: Hyperparams, stage_check: bool
    ) -> list[float]:
        ...

    def fit_model(self, X: Any, y: Any, stage_check: bool) -> float:
        ...

    def performance(self, X: Any, y: Any, stage_check: bool) -> float:
        ...

    def permutation(self, X: Any, y: Any, stage_check: bool) -> list[float]:
        ...


class IncorrectStageException(Exception):
    def __init__(self, stage):
        self.message = f"Incorrect stage for {correct_stage.name} method."
        super().__init__(self.message)
