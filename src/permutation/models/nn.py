import tensorflow as tf


class NN:
    algorithm_name: str = "Feed Forward Neural Network"
    algorithm_type: str = "Regression"
    hparams: hparams = None

    def fit_model(self) -> None:
        ...

    def evaluate_performance(self, x: Any) -> Metric:
        ...


class NN_hparams:
    pass
