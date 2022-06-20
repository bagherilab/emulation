from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from modelprotocol import Hparams
from sklearnmodel import AbstractSKLearnModel


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
    ):
        cls.__init__(hparams=hparams)
        if hparams:
            cls._model = model_dependency(**self.hparams)
        else:
            cls._model = model_dependency()

        cls._standardization = preprocessing_dependency()

        cls._pipeline = Pipeline(
            [("scaler", self._standardization), ("mlr", self._model)]
        )

        return cls
