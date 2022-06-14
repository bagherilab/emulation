import sklearn


class SVR:
    algorithm_name: str = "Support Vector Regression"
    algorithm_type: str = "SVR"
    hparams: hparams = None

    def fit_model(self) -> None:
        ...

    def evaluate_performance(self, x: Any) -> Metric:
        ...


class SVR_hparams:
    pass
