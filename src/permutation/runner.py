from typing import Optional

from permutation.metrics import BatchMetric, SequentialMetric
from permutation.models.modelprotocol import Model
from permutation.loader import Loader
from permutation.stage import Stage, IncorrectStageException


class Runner:
    """
    Runner class to manage training, alongside calls to the Model interface.
    """

    def __init__(self, model: Model, loader: Loader) -> None:
        """todo"""
        self.model = model
        self.loader = loader

        self.training_metrics: SequentialMetric = SequentialMetric(
            name=f"Model: {self.model.algorithm_name}", value_type="RMSE", stage=Stage.TRAIN
        )
        self.testing_metrics: SequentialMetric = SequentialMetric(
            name=f"Model: {self.model.algorithm_name}", value_type="RMSE", stage=Stage.TEST
        )

        self.cv_metrics: Optional[BatchMetric] = None
        self.permutation_metrics: list[BatchMetric] = []

        self.stage = Stage.TRAIN if self.model.hparams is None else Stage.VAL

    def cross_validation(self, K: int = 10) -> None:
        """todo"""
        self.stage_check(Stage.VAL)
        X, y = self.loader.load_training_data()
        self.cv_metrics = self.model.crossval_hparams(X, y, K)

    def train(self) -> None:
        """todo"""
        self.stage_check(Stage.TRAIN)
        X, y = self.loader.load_training_data()
        metric = self.model.fit_model(X, y)
        self.training_metrics.update(metric, len(X.index))

    def test(self) -> None:
        """todo"""
        self.stage_check(Stage.TEST)
        X, y = self.loader.load_testing_data()
        metric = self.model.performance(X, y)
        self.testing_metrics.update(metric, len(X.index))

    def permutation_testing(self) -> None:
        """todo"""
        self.stage_check(Stage.PERM)
        X, y = self.loader.load_working_data()
        metrics_list = self.model.permutation(X, y)
        self.permutation_metrics.extend(metrics_list)

    def set_stage(self, stage: Stage):
        self.stage = stage

    def stage_check(self, correct_stage: Stage) -> None:
        """todo"""
        if self.stage is not correct_stage:
            raise IncorrectStageException(correct_stage)

    @property
    def name(self) -> str:
        """
        name attribute to use as naming convention (file names, structure) for associated model
        """

        if self.model.hparams:
            temp_list = [f"{param}-{val}" for param, val in self.model.hparams.as_dict.items()]
            return f"{self.model.abv}__" + "_".join(temp_list)
        else:
            return self.model.abv
