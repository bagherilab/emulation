from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel


class SVReg(AbstractSKLearnModel):
    """todo"""

    algorithm_name = "Support Vector Regression"
    algorithm_type = "SVR"

    @classmethod
    def set_model(
        cls,
        model_dependency=SVR,
        hparams=None,
        preprocessing_dependencies=[("scaler", StandardScaler)],
    ) -> Model:
        """todo"""

        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )
