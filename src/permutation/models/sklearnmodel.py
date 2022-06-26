from abc import ABC, abstractmethod
from typing import Optional, Callable

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
import pandas as pd

from permutation.metrics import BatchMetric, SequentialMetric
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams


class AbstractSKLearnModel(ABC):
    algorithm_name: str
    algorithm_type: str
    hparams: Hyperparams
    _pipeline: Pipeline

    @classmethod
    @abstractmethod
    def set_model(
        cls: Model,
        model_dependency: BaseEstimator,
        preprocessing_dependency: Optional[TransformerMixin],
    ) -> Model:
        ...

    def crossval_hparams(
        self, X: pd.DataFrame, y: pd.Series, K: int = 10
    ) -> BatchMetric:
        metrics = BatchMetric(f"R^2 Cross Validation - K = {K}")
        cv_results = cross_val_score(self._pipeline, X, y, cv=K)
        metrics.batchupdate(cv_results.tolist())
        return metrics

    def permutation(
        self, X: pd.DataFrame, y: pd.Series, repeats=30
    ) -> list[BatchMetric]:
        self._pipeline.fit(X, y)
        perm_output = permutation_importance(self._pipeline, X, y, repeats=30)
        list_of_metrics = parse_permutation_output(perm_output, X.columns.to_list())
        return list_of_metrics

    def fit_model(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        self._pipeline.fit(X, y)
        return self.performance(X, y)

    def performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = self._pipeline.predict(X)
        return RMSE(y, y_pred)


def parse_permutation_output(
    output: Bunch, feature_names: list[str]
) -> list[BatchMetric]:
    metric_list = []
    for i, feature in enumerate(feature_names):
        temp_metric = BatchMetric(name=f"R^2, Feature: {feature}")
        importance_val_list = output.importances[:, i].flatten().tolist()
        temp_metric.batchupdate(importance_val_list)
        metric_list.append(temp_metric)
    return metric_list


def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)
