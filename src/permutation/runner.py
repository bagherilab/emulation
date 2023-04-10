from typing import Optional
import uuid

import pandas as pd

from permutation.metrics import BatchMetric
from permutation.models.modelprotocol import Model
from permutation.loader import Loader
from permutation.stage import Stage, IncorrectStageException


class Runner:
    """
    Runner class to manage training, alongside calls to the Model interface.

    Attributes
    ----------
    model :
        The Model that is being controlled by the runner
    loader :
        The Loader object that holds the train/test splits and controls subsampling
    cv_metrics :
        Metric object that holds the cross-validation performance
    training_metrics :
        Metric object that hold the RMSE from training data
    testing_metrics :
        Metric object that hold the RMSE from testing data
    permutation_metrics :
        List of Metric objects that hold the R^2 from permutation testing
    name :
        Naming convention determined by:
        <modelabbreviation>_n=<observations>__<hyperparameter>=<value>

    Methods
    -------
    cross_validation(K):
        Performs and stores the <K>-fold cross validation

    train():
        Train the associated model using the training data from loader

    test():
        Test the trained model using the withheld test data

    permutation_testing():
        Perform permutation testing on trained model

    set_stage(stage):
        Allows other objects to update stage

    stage_check(correct_stage):
        Raises an exception if stage not the same as the passed correct_stage

    reset():
        Resets all the metric objects to empty state, sets stage back to initialization value
    """

    def __init__(self, model: Model, loader: Loader) -> None:
        self.model = model
        self.loader = loader

        self.training_metrics: BatchMetric = BatchMetric(
            name=f"Model: {self.model.algorithm_name}",
            value_type="R^2",
            stage=Stage.TRAIN,
        )
        self.testing_metrics: BatchMetric = BatchMetric(
            name=f"Model: {self.model.algorithm_name}",
            value_type="R^2",
            stage=Stage.TEST,
        )

        self.cv_metrics: Optional[BatchMetric] = None
        self.permutation_metrics: list[BatchMetric] = []

        self._stage = Stage.TRAIN if self.model.hparams is None else Stage.VAL
        self._UUID = uuid.uuid4()

    def cross_validation(self, K: int = 10) -> None:
        """Performs and stores the <K>-fold cross validation"""
        self.stage_check(Stage.VAL)
        X, y = self.loader.load_training_data()
        self.cv_metrics = self.model.crossval_hparams(X, y, K)

    def train(self) -> None:
        """Train the associated model using the training data from loader"""
        self.stage_check(Stage.TRAIN)
        X, y = self.loader.load_training_data()
        metric = self.model.fit_model(X, y)
        self.training_metrics.update(metric)
        self.training_metrics.num_observations.append(len(y))

    def test(self) -> None:
        """Test the trained model using the withheld test data"""
        self.stage_check(Stage.TEST)
        X, y = self.loader.load_testing_data()
        metric = self.model.performance(X, y)
        self.testing_metrics.update(metric)
        self.testing_metrics.num_observations.append(len(y))

    def permutation_testing(self) -> None:
        """Perform permutation testing on trained model"""
        self.stage_check(Stage.PERM)
        X, y = self.loader.load_working_data()
        metrics_list = self.model.permutation(X, y)
        self.permutation_metrics.extend(metrics_list)

    def get_predictions(self) -> pd.DataFrame:
        """Get predictions from the triaed model"""
        X, y = self.loader.load_working_data()
        df = pd.DataFrame(self.model.get_predicted_values(X), columns=["y_pred"], index=y.index)
        df["set"] = ""
        df.loc[self.loader._training_idx, "set"] = "train"
        df.loc[self.loader._testing_idx, "set"] = "test"
        return df

    def set_stage(self, stage: Stage) -> None:
        """Allows other objects to update stage"""
        self._stage = stage

    def stage_check(self, correct_stage: Stage) -> None:
        """Raises an exception if stage not the same as the passed correct_stage"""
        if self._stage is not correct_stage:
            raise IncorrectStageException(self._stage, correct_stage)

    def reset(self) -> None:
        """
        Resets all the metric objects to empty state,
        sets stage back to initialization value
        """
        self.training_metrics = BatchMetric(
            name=f"Model: {self.model.algorithm_name}", value_type="RMSE", stage=Stage.TRAIN
        )
        self.testing_metrics = BatchMetric(
            name=f"Model: {self.model.algorithm_name}", value_type="RMSE", stage=Stage.TEST
        )
        self.cv_metrics = None
        self.permutation_metrics = []

        self._stage = Stage.TRAIN if self.model.hparams is None else Stage.VAL

    @property
    def id(self) -> str:
        """Return UUID"""
        return str(self._UUID)

    @property
    def stage(self) -> Stage:
        """Return current running stage"""
        return self._stage

    def reset_id(self) -> None:
        """Generates a new random unique ID"""
        self._UUID = uuid.uuid4()

    @property
    def description(self) -> dict[str, str]:
        """Returns a dictionary containing information about the current model"""
        return_dict = {"model_type": self.model.algorithm_abv, "n": str(self.loader.n_working)}
        if self.model.hparams:
            return_dict.update(self.model.hparams.as_dict())
        return return_dict
