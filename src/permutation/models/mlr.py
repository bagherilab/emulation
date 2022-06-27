from typing import Optional, Tuple, List

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
    _standardization: Optional[List[Tuple[str, TransformerMixin]]] = None
    _pipeline: Pipeline = None

    @classmethod
    def set_model(
        cls,
        hparams=None,
        preprocessing_dependencies=("scaler", StandardScaler),
        model_dependency=ElasticNet,
    ) -> Model:
        model = cls()
        model.hparams = hparams

        pipeline_list = []

        if preprocessing_dependencies:
            model._standardization = [
                (name, package()) for name, package in preprocessing_dependencies
            ]
            pipeline_list.extend(model._standardization)

        if hparams:
            model.algorithm_name += f", hparams: {hparams}"
            model._model = model_dependency(**hparams.as_dict())
        else:
            model._model = model_dependency()

        pipeline_list.append(("mlr", model._model))

        model._pipeline = Pipeline(pipeline_list)
        return model
