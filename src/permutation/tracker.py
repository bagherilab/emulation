from pathlib import Path
from typing import Optional

from permutation.runner import Runner
from permutation.loader import Loader, CSVLoader
from permutation.models.hyperparameters import Hyperparams
from permutation.models.modelprotocol import Models


class ExperimentTracker:
    """todo"""

    def __init__(
        self,
        log_dir: str,
        data_path: str,
    ) -> None:
        """todo"""

        _validate_log_dir(log_dir)
        self.logger: Logger = Logger(log_dir)
        self.loader: Loader = CSVLoader(data_path)
        self._models: Dict[str, list[Runner]] = {}

    def add_model(self, model: Model) -> None:
        """todo"""
        self._check_algorithm_in_models(model.algorithm_name)
        self._models[model.algorithm_name].append(Runner(model, self.loader))

    def _check_algorithm_in_models(self, name) -> None:
        if name not in self._models:
            self._models[name] = []

    @property
    def models(self) -> list[str]:
        list_of_models = []
        for _, runner_list in self._models.items():
            temp_name_list = [runner.name for runner in runner_list]
            list_of_models.extend(temp_summary_list)
        return list_of_models

    @property
    def algorithms(self):
        return self._models.keys

    def hyperparameter_selection(self) -> None:
        """todo"""

    def train_models(self) -> None:
        """todo"""

    def permutation_testing(self) -> None:
        """todo"""

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True) -> None:
        """todo"""
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        if not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")
