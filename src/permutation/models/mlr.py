from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import Hparams
from permutation.models.sklearnmodel import AbstractSKLearnModel


class MLR(AbstractSKLearnModel):
    algorithm_name: str = "Regularized Linear Regression"
    algorithm_type: str = "Regression"
    hparams: Hparams = None
    _model: BaseEstimator = None
    _standardization: TransformerMixin = None
    _pipeline: Pipeline = None

    @classmethod
    def set_model(
        cls,
        hparams=None,
        preprocessing_dependency=StandardScaler,
        model_dependency=ElasticNet,
    ) -> Model:
        model = cls()
        model.hparams = hparams

        if hparams:
            model.algorithm_name += f", hparams: {hparams}"
            model._model = model_dependency(**hparams.as_dict())
        else:
            model._model = model_dependency()

        if preprocessing_dependency:
            model._standardization = preprocessing_dependency()

            model._pipeline = Pipeline(
                [("scaler", model._standardization), ("mlr", model._model)]
            )
            return model

        model._pipeline = Pipeline([("mlr", model._model)])
        return model
