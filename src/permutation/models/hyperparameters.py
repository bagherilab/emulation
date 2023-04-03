from typing import Optional, Any

from permutation.metrics import BatchMetric


class HParams:
    """Dataclass for storing hyperparameter information and associated performance metrics."""

    def __init__(self, param_dict: Optional[dict[str, Any]] = None):
        self.args: list[str] = []
        self.values: list[Any] = []
        self.number_of_parameters: int = 0
        self.cross_validation_performance: Optional[BatchMetric] = None
        self.performance_average: float = 0.0

        if param_dict is not None:
            self.update_params(param_dict)

    def update_params(self, param_dict: dict[str, Any]) -> None:
        """Update a dictionary of parameters with new values"""
        args, values = zip(*param_dict.items())
        for arg, value in zip(args, values):
            self.update_param(arg, value)

    def update_param(self, arg: str, value: Any) -> None:
        """Update a single parameter with a new value"""
        if self.cross_validation_performance:
            raise AttributeError("Already performed cross-validation with these parameters.")
        self.args.append(arg)
        self.values.append(value)
        self.number_of_parameters += 1

    def as_dict(self) -> dict[str, Any]:
        """Return the hyperparameters and values as a dict"""
        return dict(zip(self.args, self.values))

    def __str__(self) -> str:
        """String representation of hyperparameters as a dict"""
        return str(self.as_dict())
