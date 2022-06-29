from dataclasses import dataclass, field
from typing import Protocol, Optional, List, Any

from permutation.stage import Stage
from permutation.metrics import BatchMetric


class Hyperparams(Protocol):
    """Protocol for typing hyperparameter object."""

    args: str
    values: str
    number_of_parameters: int

    def update_params(self) -> None:
        ...

    def as_dict(self, stage: Stage) -> dict:
        ...

    def update_cv_metrics(self, value: float) -> None:
        ...


@dataclass
class Hparams:
    """Dataclass for storing hyperparameter information and associated performance metrics."""

    args: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    number_of_parameters: int = 0
    cross_validation_performance: Optional[BatchMetric] = None
    performance_average: float = 0.0

    def update_params(self, arg, value) -> None:
        """todo"""
        if self.cross_validation_performance:
            raise AttributeError("Already performed cross-validation with these parameters.")
        self.args.append(arg)
        self.values.append(value)
        self.number_of_parameters += 1

    def as_dict(self) -> None:
        """todo"""
        return dict(zip(self.args, self.values))

    def update_cv_metrics(self, values: list[float]) -> None:
        """todo"""
        if self.cross_validation_performance is None:
            self.cross_validation_performance = BatchMetric(str(self))

        self.cross_validation_performance.batchupdate(values)
        self.performance_average = self.cross_validation_performance.average

    def __str__(self) -> str:
        """todo"""
        return str(self.as_dict())
