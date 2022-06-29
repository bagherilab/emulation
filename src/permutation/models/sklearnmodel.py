from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Iterable

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
import pandas as pd

from permutation.metrics import BatchMetric
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hyperparams


class AbstractSKLearnModel(ABC):
    """todo"""

    algorithm_name: str
    algorithm_type: str
    pipeline: Pipeline
    model: BaseEstimator
    hparams: Optional[Hyperparams] = None
    standarization: Optional[Iterable[Tuple[str, TransformerMixin]]] = None

    @classmethod
    @abstractmethod
    def set_model(
        cls: Model,
        model_dependency: BaseEstimator,
        hparams: Optional[Hyperparams],
        preprocessing_dependencies: Optional[Iterable[Tuple[str, TransformerMixin]]],
    ) -> Model:
        """todo"""
        ...

    @classmethod
    def _set_model(
        cls,
        model_dependency: BaseEstimator,
        hparams: Optional[Hyperparams] = None,
        preprocessing_dependencies: Optional[Iterable[Tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """todo"""
        model = cls()
        model.hparams = hparams

        pipeline_list = []

        if preprocessing_dependencies:
            model.standarization = [
                (name, package()) for name, package in preprocessing_dependencies
            ]
            pipeline_list.extend(model.standarization)

        if hparams:
            model.algorithm_name += f", hparams: {hparams}"
            model.model = model_dependency(**hparams.as_dict())
        else:
            model.model = model_dependency()

        pipeline_list.append((model.algorithm_name, model.model))

        model.pipeline = Pipeline(pipeline_list)
        return model

    def crossval_hparams(self, X: pd.DataFrame, y: pd.Series, K: int = 10) -> BatchMetric:
        """todo"""
        metrics = BatchMetric(f"R^2 Cross Validation - K = {K}")
        cv_results = cross_val_score(self.pipeline, X, y, cv=K)
        metrics.batchupdate(cv_results.tolist())
        return metrics

    def permutation(self, X: pd.DataFrame, y: pd.Series, repeats: int = 30) -> List[BatchMetric]:
        """todo"""
        self.pipeline.fit(X, y)
        perm_output = permutation_importance(self.pipeline, X, y, n_repeats=repeats)
        list_of_metrics = parse_permutation_output(perm_output, X.columns.to_list())
        return list_of_metrics

    def fit_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """todo"""
        self.pipeline.fit(X, y)
        return self.performance(X, y)

    def performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """todo"""
        y_pred = self.pipeline.predict(X)
        return root_mean_square_error(y, y_pred)


def parse_permutation_output(output: Bunch, feature_names: List[str]) -> List[BatchMetric]:
    """todo"""
    metric_list = []
    for i, feature in enumerate(feature_names):
        temp_metric = BatchMetric(name=f"R^2, Feature: {feature}")
        importance_val_list = output.importances[:, i].flatten().tolist()
        temp_metric.batchupdate(importance_val_list)
        metric_list.append(temp_metric)
    return metric_list


def root_mean_square_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """todo"""
    return mean_squared_error(y_true, y_pred, squared=False)
