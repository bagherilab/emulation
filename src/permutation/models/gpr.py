from typing import Optional, Iterable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, DotProduct
from sklearn.base import BaseEstimator, TransformerMixin

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel
from permutation.models.hyperparameters import HParams
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, DotProduct

def map_kernel(kernel_name: str):
    """
    Map kernel name to an actual kernel object dynamically.
    
    Parameters:
    - kernel_name (str): The name of the kernel as defined in the configuration.

    Returns:
    - kernel (Kernel): The corresponding kernel object.

    Raises:
    - ValueError: If the kernel_name is not recognized.
    """
    if kernel_name == "DotProduct + WhiteKernel":
        return DotProduct() + WhiteKernel()
    elif kernel_name == "RBF":
        return RBF(length_scale=1.0)
    elif kernel_name == "Matern":
        return Matern(nu=1.5)
    elif kernel_name == "RationalQuadratic":
        return RationalQuadratic(alpha=1.0, length_scale=1.0)
    elif kernel_name == "Exponential":
        return RBF(length_scale=1.0)  # Exponential is a special case of RBF
    else:
        raise ValueError(
            f"Invalid kernel name '{kernel_name}'. Valid options are: "
            "'DotProduct + WhiteKernel', 'RBF', 'Matern', 'RationalQuadratic', 'Exponential'."
        )


class GPReg(AbstractSKLearnModel):
    """
    Gaussian Process Regression model

    Methods
    -------
    set_model(cls, model_dependency, hparams, preprocessing_dependencies, kernel_code):
        Set up the model, preprocessing pipeline, and read in hyperparameters, including kernel configuration.
    """

    algorithm_name = "Gaussian Process Regression"
    algorithm_abv = "GPR"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        model_dependency: BaseEstimator = GaussianProcessRegressor,
        hparams: Optional[HParams] = None,
        preprocessing_dependencies: Optional[Iterable[tuple[str, TransformerMixin]]] = None,
    ) -> Model:
        """Set up model from config files and super class"""
        if preprocessing_dependencies is None:
            preprocessing_dependencies = []

        kernel = None

        if hparams:
            hparams_dict = hparams.as_dict()
            kernel_name = hparams_dict.get("kernel")
            kernel = map_kernel(kernel_name)
            hparams_dict['kernel'] = kernel
            hparams = HParams(param_dict=hparams_dict)
        else:
            hparams_dict = {'kernel': kernel} if kernel else {}

        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )
