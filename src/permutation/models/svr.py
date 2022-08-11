from typing import Optional, Iterable

from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel
from permutation.models.hyperparameters import Hyperparams


class SVReg(AbstractSKLearnModel):
    """todo"""

    algorithm_name = "Support Vector Regression"
    algorithm_abv = "SVR"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        model_dependency: BaseEstimator = SVR,
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
