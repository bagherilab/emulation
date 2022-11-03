from permutation.models.mlr import MLR
from permutation.models.rf import RF
from permutation.models.svr import SVReg
from permutation.models.mlp import MLP
from permutation.models.modelprotocol import Model
from permutation.models.hyperparameters import HParams

MODEL_DEPENDENCIES = {"mlr": MLR, "rf": RF, "svr": SVReg, "mlp": MLP}


def _get_correct_model(model_type: str, hparams: HParams) -> Model:
    """Choose the correct model dependency from model string"""
    try:
        return MODEL_DEPENDENCIES[model_type].set_model(hparams=hparams)
    except KeyError:
        raise ValueError(f"Model type {model_type} is not implemented.")


def assign_models_from_list(hparams_list: list[HParams], model_type: str) -> list[Model]:
    """Create list of model instances"""
    return [_get_correct_model(model_type, hparams) for hparams in hparams_list]
