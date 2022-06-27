from pathlib import Path
from typing import Optional, List

from permutation.runner import Runner
from permutation.models.hyperparameters import Hyperparams


class ExperimentTracker:
    """todo"""

    def __init__(
        self,
        log_dir: str,
        runner: Runner,
        hparam_set: Optional[List[Hyperparams]] = None,
    ):
        """todo"""

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        """todo"""
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        if not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")
