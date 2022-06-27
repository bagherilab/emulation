from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from permutation.models.sklearnmodel import AbstractSKLearnModel


class SVR(AbstractSKLearnModel):
    algorithm_name: str = "Support Vector Regression"
    algorithm_type: str = "SVR"

    @classmethod
    def set_model(
        cls,
        hparams=None,
        preprocessing_dependencies=[("scaler", StandardScaler)],
        model_dependency=SVR,
    ) -> Model:
        return super()._set_model(
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
            model_dependency=model_dependency,
        )
