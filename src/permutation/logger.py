import pandas as pd

from permutation.metrics import Metric


class Logger:
    log_path: str | Path

    def metric_to_csv(experiment: str, model: str, metric: Metric) -> None:
        """todo"""
        self.pandas_to_csv(experiment, model, metric.name, metric.stage, metric.to_pandas())

    def pandas_to_csv(
        experiment: str, model: str, filename: str, stage: Stage, df: pd.DataFrame | pd.Series
    ) -> None:
        path = f"{self.log_path}/{experiment}/{model}/{stage}/{filename}.csv"
        df.to_csv(path)
