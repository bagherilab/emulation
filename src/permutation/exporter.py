from pathlib import Path
import json

import pandas as pd

from permutation.metrics import Metric
from permutation.runner import Runner
from permutation.stage import Stage
from permutation.file_utils import validate_dir


class Exporter:
    """
    A class for exporting experimental results to disk

    Attributes
    ----------
    export_path :
        Location where files will be stored
    experiment :
        Name of experiment

    Methods
    -------
    metric_to_csv(experiment, model, filename, metric):
        Takes in arguments to write to csv file using filenaming and path conventions,
        using Metric methods

    pandas_to_csv(experiment, model, filename, stage, dataframe):
        Writes data to csv file from pandas Dataframe using filenaming and path conventions
    """

    def __init__(self, experiment: str, export_path: str = "/results") -> None:
        self.export_path = export_path
        self.experiment = experiment
        validate_dir(self.export_path)

    def metric_to_csv(self, model: str, filename: str, metric: Metric) -> None:
        """Takes in arguments to write to csv file, using Metric methods"""
        self.pandas_to_csv(model, filename, metric.stage.name, metric.to_pandas())

    def pandas_to_csv(
        self,
        model: str,
        filename: str,
        stage: Stage,
        dataframe: pd.DataFrame | pd.Series,
    ) -> None:
        """Writes data to csv file from pandas Dataframe using filenaming and path conventions"""
        dir_path = f"{self.export_path}/{self.experiment}/{model}/{stage}"
        self._save_df(dir_path, filename, dataframe)

    def _save_df(self, dir_path, name, df: pd.DataFrame):
        """Validates save path and then saves a Dataframe as a CSV file"""
        validate_dir(dir_path)
        file_path = f"{dir_path}/{name}.csv"
        df.to_csv(file_path)

    def save_manifest_file(self, manifest: pd.DataFrame):
        """Saves the manifest as a CSV under the correct experiment"""
        dir_path = f"{self.export_path}{self.experiment}/"
        self._save_df(dir_path, "manifest", manifest)

    def save_model_json(self, runner: Runner):
        """Saves the model parameters as a JSON file"""
        dir_path = (
            f"{self.export_path}{self.experiment}/{runner.model.algorithm_abv}/{runner.id}.json"
        )
        with open(dir_path, "w") as outfile:
            json.dump(runner.model.hparams.as_dict(), outfile)

    def save_predictions(self, model: str, runner: Runner):
        """Saves the model predictions as a CSV file"""
        dir_path = f"{self.export_path}{self.experiment}/{model}/"
        predictions = runner.get_predictions()
        self._save_df(dir_path, f"{runner.id}.PREDICTIONS", predictions)

    def save_train_test(self, train: pd.DataFrame, test: pd.DataFrame):
        """Saves the model predictions as a CSV file"""
        train_dir_path = f"{self.export_path}{self.experiment}"
        test_dir_path = f"{self.export_path}{self.experiment}"
        self._save_df(train_dir_path, "train", train)
        self._save_df(test_dir_path, "test", test)
