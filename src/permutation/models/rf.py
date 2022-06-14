import sklearn


class RF:
    algorithm_name: str = "Random Forest"
    algorithm_type: str = "Regression"
    hparams: hparams = None

    def fit_model(self) -> None:
        ...

    def evaluate_performance(self, x: Any) -> Metric:
        ...


class RF_hparams:
    pass
