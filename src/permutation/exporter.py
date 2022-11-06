from pathlib import Path
import json

import pandas as pd

from permutation.metrics import Metric
from permutation.runner import Runner
from permutation.stage import Stage
from permutation.file_utils import validate_dir


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

    def __init__(self, experiment: str, export_path: str = "/results") -> None:
        self.export_path = export_path
        self.experiment = experiment
        validate_dir(self.export_path)

    def metric_to_csv(self, experiment: str, model: str, filename: str, metric: Metric) -> None:
        """takes in arguments to write to csv file, using Metric methods"""
        self.pandas_to_csv(model, filename, metric.stage, metric.to_pandas())

    def pandas_to_csv(
        self,
        model: str,
        filename: str,
        stage: Stage,
        dataframe: pd.DataFrame | pd.Series,
    ) -> None:
        """writes data to csv file from pandas Dataframe using filenaming and path conventions"""
        dir_path = f"{self.export_path}/{self.experiment}/{model}/{stage}"
        self._save_df(dir_path, filename, dataframe)

    def _save_df(self, dir_path, name, df: pd.DataFrame):
        validate_dir(dir_path)
        file_path = f"{dir_path}/{name}.csv"
        df.to_csv(file_path)

    def save_manifest_file(self, manifest: pd.DataFrame):
        dir_path = f"{self.export_path}{self.experiment}/"
        self._save_df(dir_path, "manifest", manifest)

    def save_model_json(self, runner: Runner):
        dir_path = (
            f"{self.export_path}{self.experiment}/{runner.model.algorithm_abv}/{runner.id}.json"
        )
        with open(dir_path, "w") as outfile:
            json.dump(runner.model.hparams.as_dict(), outfile)
