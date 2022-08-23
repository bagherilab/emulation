from pathlib import Path

import pandas as pd

from permutation.metrics import Metric
from permutation.stage import Stage
from permutation.file_util import validate_dir


class Exporter:
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
    """

    def __init__(self, export_path: str) -> None:
        self.export_path = export_path
        validate_dir(self.export_path)

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
        dir_path = f"{self.export_path}/{experiment}/{model}/{stage}"
        validate_dir(dir_path)
        file_path = f"{dir_path}/{filename}.csv"
        dataframe.to_csv(file_path)
