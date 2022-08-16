import pandas as pd

from permutation.metrics import Metric


class Logger:
    log_path: str | Path

    def metric_to_csv(experiment: str, model: str, filename: str, metric: Metric) -> None:
        """todo"""
        self.pandas_to_csv(experiment, model, filename, metric.stage, metric.to_pandas())

    def pandas_to_csv(
        experiment: str, model: str, filename: str, stage: Stage, df: pd.DataFrame | pd.Series
    ) -> None:
        path = f"{self.log_path}/{experiment}/{model}/{stage}/{filename}.csv"
        df.to_csv(path)

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
