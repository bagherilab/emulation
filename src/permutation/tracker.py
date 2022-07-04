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
        self.logger: Logger = Logger(log_dir)
        self.loader: Loader = CSVLoader(data_path)
        self._models: List[Models] = []

    def add_model(self, model: Model) -> None:
        """todo"""
        self._models.append(model)

    def hyperparameter_selection(self) -> None:
        """todo"""
        for model in self._models():
            runner = Runner(model, self.loader)

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
