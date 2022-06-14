import sklearn as sk


class MLR:
    algorithm_name: str = "Regularized Linear Regression"
    algorithm_type: str = "Regression"
    hparams: hparams = None

    def fit_model(self, x: Any) -> None:
        ...

    def evaluate_performance(self, x: Any) -> Metric:
        ...


class MLR_hparams:
    pass
