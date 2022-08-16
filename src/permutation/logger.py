from pathlib import Path

import pandas as pd

from permutation.metrics import Metric
from permutation.stage import Stage


class Logger:
    """
    Attributes
    ----------
    log_path :
        location where files will be stored

    Methods
    -------
    metric_to_csv(experiment, model, filename, metric):
        takes in arguments to write to csv file using filenaming and path conventions,
        using Metric methods

    pandas_to_csv(experiment, model, filename, stage, dataframe):
        writes data to csv file from pandas Dataframe using filenaming and path conventions

    TODO: Add log file implementation
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._validate_log_dir(self.log_path)

    def metric_to_csv(self, experiment: str, model: str, filename: str, metric: Metric) -> None:
        """takes in arguments to write to csv file, using Metric methods"""
        self.pandas_to_csv(experiment, model, filename, metric.stage, metric.to_pandas())

    def pandas_to_csv(
        self,
        experiment: str,
        model: str,
        filename: str,
        stage: Stage,
        dataframe: pd.DataFrame | pd.Series,
    ) -> None:
        """writes data to csv file from pandas Dataframe using filenaming and path conventions"""
        dir_path = f"{self.log_path}/{experiment}/{model}/{stage}"
        self._validate_log_dir(dir_path)
        file_path = f"{dir_path}/{filename}.csv"
        dataframe.to_csv(file_path)

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True) -> None:
        """
        validates the directory exists and creates if not,
        or raises an error if create is false
        """
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        if not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")
