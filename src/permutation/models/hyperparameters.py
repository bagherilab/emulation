from dataclasses import dataclass, field
from typing import Protocol, Optional, Any

from permutation.metrics import BatchMetric


class Hyperparams(Protocol):
    """Protocol for typing hyperparameter object."""

    args: str
    values: str
    number_of_parameters: int

    def update_params(self) -> None:
        """todo"""

    def as_dict(self) -> dict[str, Any]:
        """todo"""

    def update_cv_metrics(self, value: float) -> None:
        """todo"""


@dataclass
class Hparams:
    """Dataclass for storing hyperparameter information and associated performance metrics."""

    args: list[str] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    number_of_parameters: int = 0
    cross_validation_performance: Optional[BatchMetric] = None
    performance_average: float = 0.0

    def update_params(self, arg: str, value: Any) -> None:
        """todo"""
        if self.cross_validation_performance:
            raise AttributeError("Already performed cross-validation with these parameters.")
        self.args.append(arg)
        self.values.append(value)
        self.number_of_parameters += 1

    def as_dict(self) -> dict[str, Any]:
        """todo"""
        return dict(zip(self.args, self.values))

    def __str__(self) -> str:
        """todo"""
        return str(self.as_dict())
