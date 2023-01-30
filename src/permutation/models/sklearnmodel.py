from abc import ABC, abstractmethod
from typing import Optional, Iterable

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
import pandas as pd
import numpy as np

from permutation.metrics import BatchMetric
from permutation.stage import Stage
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import HParams


class AbstractSKLearnModel(ABC):
    """Abstract class for sklearn based models."""

    algorithm_name: str
    algorithm_type: str
    algorithm_abv: str
    pipeline: Pipeline
    model: BaseEstimator
    hparams: Optional[HParams] = None
    standarization: Optional[Iterable[tuple[str, TransformerMixin]]] = None

    @classmethod
    @abstractmethod
    def set_model(
        cls: Model,
        model_dependency: BaseEstimator,
        hparams: Optional[HParams],
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]],
    ) -> Model:
        """Abstract method for to implement models with sklearn defaults"""

    @classmethod
    def _set_model(
        cls,
        model_dependency: BaseEstimator,
        hparams: Optional[HParams] = None,
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """
        Helper function for subclasses:
        Set up pipeline for sklearn to allow for flexibility in preprocessing steps
        """
        model = cls()
        model.hparams = hparams

        pipeline_list = []

        if preprocessing_dependencies:
            model.standarization = [
                (name, package()) for name, package in preprocessing_dependencies
            ]
            pipeline_list.extend(model.standarization)

        if hparams:
            model.algorithm_name += f", hparams: {hparams}"  # pylint: disable=no-member
            model.model = model_dependency(**hparams.as_dict())
        else:
            model.model = model_dependency()

        pipeline_list.append((model.algorithm_name, model.model))  # pylint: disable=no-member

        model.pipeline = Pipeline(pipeline_list)
        return model

    def crossval_hparams(self, X: pd.DataFrame, y: pd.Series, K: int = 10) -> BatchMetric:
        """Perform cross validation"""
        metrics = BatchMetric(name=f"CV_K={K},", value_type="R^2", stage=Stage.VAL)
        try:
            cv_results = cross_val_score(self.pipeline, X, y, cv=K)
        except ValueError:
            raise ValueError(f"Hparams are {self.hparams}")
        metrics.batchupdate(cv_results.tolist())
        return metrics

    def fit_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Fit the model according to sklearn functionality"""
        self.pipeline.fit(X, y)
        return self.performance(X, y)

    def performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate performance of model by comparing predicted values to known"""
        return self.pipeline.score(X, y)

    def get_predicted_values(self, X: pd.DataFrame) -> np.ndarray | pd.Series:
        """Predict y from X"""
        return self.pipeline.predict(X)

    def permutation(self, X: pd.DataFrame, y: pd.Series, repeats: int = 30) -> list[BatchMetric]:
        """Perform permutation testing"""
        self.pipeline.fit(X, y)
        perm_output = permutation_importance(self.pipeline, X, y, scoring="r2", n_repeats=repeats)
        list_of_metrics = parse_permutation_output(perm_output, feature_names=X.columns.to_list())
        return list_of_metrics


def parse_permutation_output(output: Bunch, feature_names: list[str]) -> list[BatchMetric]:
    """Helper function for extracting from sklearn Bunch object for permutation testing"""
    metric_list = []
    for i, feature in enumerate(feature_names):
        temp_metric = BatchMetric(name=f"Feature:{feature}", value_type="R^2", stage=Stage.PERM)
        importance_val_list = output.importances[i, :].flatten().tolist()
        temp_metric.batchupdate(importance_val_list)
        metric_list.append(temp_metric)
    return metric_list


def root_mean_square_error(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Find the RMSE of linked data collections"""
    return mean_squared_error(y_true, y_pred, squared=False)
