from typing import Optional, Iterable
from ast import literal_eval

from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel
from permutation.models.hyperparameters import HParams


class MLP(AbstractSKLearnModel):
    """
    Multi-Layer Perceptron model

    Methods
    -------
    set_model(cls, model_dependency, hparams, preprocessing_dependencies):
        Set up the model, preprocessing pipeline and read in hyperparameters
    """

    algorithm_name = "Multi-layer Perceptron Regressor"
    algorithm_abv = "MLP"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        model_dependency: BaseEstimator = MLPRegressor,
        hparams: Optional[HParams] = None,
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """Set up model from config files and super class"""
        if preprocessing_dependencies is None:
            preprocessing_dependencies = []

        if hparams is not None:
            _fix_tuple_type(hparams)

        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )


def _fix_tuple_type(
    hparams: HParams,
    change_arg: str = "hidden_layer_sizes",
) -> None:
    """Makes sure config parameter value is evaluated as correct type"""
    for i, arg_tuple in enumerate(zip(hparams.args, hparams.values)):
        arg, val = arg_tuple
        if arg == change_arg:
            hparams.values[i] = literal_eval(val)
