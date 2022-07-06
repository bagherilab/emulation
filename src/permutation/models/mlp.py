from typing import Optional, Iterable

from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel
from permutation.models.hyperparameters import Hyperparams


class MLP(AbstractSKLearnModel):
    """todo"""

    algorithm_name = "Multi-layer Perceptron Regressor"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        model_dependency: BaseEstimator = MLPRegressor,
        hparams: Optional[Hyperparams] = None,
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """todo"""
        if preprocessing_dependencies is None:
            preprocessing_dependencies = []

        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )
