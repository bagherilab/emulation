import logging
import os

from permutation.file_utils import validate_dir


class Logger(ABC):
    experiment_name: str
    log_dir: str

    @abstractmethod
    def log(message: str) -> None:
        """abstract method to implement for logger objects"""


class ExperimentLogger:
    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[str],
        format_str: str = "%(asctime)s:%(name)s:%(message)s",
        level: int = logging.INFO,
    ):
        self.log_path = os.path.join(f"{self.log_dir}/{self.experiment_name}.log")
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(name)

    def _set_up(self, format_str: str, level: int):
        """todo"""
        self.logger.setLevel(level)
        formatter = logging.Formatter(format_str)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, message: str) -> None:
        """todo"""
        self.logger.info(message)
