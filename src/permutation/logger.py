from permutation.metrics import Metric


class Logger:
    log_path: str | Path

    def metric_to_csv(experiment: str, metric: Metric) -> None:
        """todo"""
        path = f"{self.path}/{experiment}/{metric.stage}/{metric.name}.csv"
        metric.to_pandas().to_csv(path)
