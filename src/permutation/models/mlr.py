from typing import Optional, Iterable

from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator, TransformerMixin

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel
from permutation.models.hyperparameters import HParams


class MLR(AbstractSKLearnModel):
    """
    Linear Regression model
        
    Methods
    -------
    set_model(cls, model_dependency, hparams, preprocessing_dependencies):
        Set up the model, preprocessing pipeline and read in hyperparameters
    """

    algorithm_name = "Regularized Linear Regression"
    algorithm_abv = "MLR"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        model_dependency: BaseEstimator = ElasticNet,
        hparams: Optional[HParams] = None,
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """Set up model from config files and super class"""
        if preprocessing_dependencies is None:
            preprocessing_dependencies = []

        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )
