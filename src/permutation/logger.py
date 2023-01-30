import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from permutation.file_utils import validate_dir


class Logger(ABC):
    experiment_name: str
    log_dir: str

    @abstractmethod
    def log(message: str) -> None:
        """Abstract method to implement for logger objects"""


class ExperimentLogger:
    """
    A class for logging experimental details

    Attributes
    ----------
    experiment_name :
        Name of the experiment
    log_dir :
        Directory to save log files in
    format_str :
        Format of log messages
    level :
        Level of detail for logger

    Methods
    -------
    log(message):
        Log a message
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "/logs/",
        format_str: str = "%(asctime)s:%(name)s:%(message)s",
        level: int = logging.INFO,
    ):
        validate_dir(log_dir)
        self.log_path = os.path.join(f"{log_dir}/{experiment_name}.log")
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(self.experiment_name)
        self._set_up(format_str, level)

    def _set_up(self, format_str: str, level: int):
        """Sets up logger with formatting, log path, and detail leval"""
        self.logger.setLevel(level)
        formatter = logging.Formatter(format_str)
        file_handler = logging.FileHandler(self.log_path, mode="w")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, message: str) -> None:
        """Log a passed message"""
        self.logger.info(message)
