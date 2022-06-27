from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from permutation.models.sklearnmodel import AbstractSKLearnModel


class MLR(AbstractSKLearnModel):
    algorithm_name = "Regularized Linear Regression"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        hparams=None,
        preprocessing_dependencies=[("scaler", StandardScaler)],
        model_dependency=ElasticNet,
    ):
        return super()._set_model(
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
            model_dependency=model_dependency,
        )
