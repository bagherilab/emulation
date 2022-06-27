from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from permutation.models.sklearnmodel import AbstractSKLearnModel


class RF(AbstractSKLearnModel):
    algorithm_name: str = "Random Forest"
    algorithm_type: str = "Regression"

    @classmethod
    def set_model(
        cls,
        hparams=None,
        preprocessing_dependencies=[("scaler", StandardScaler)],
        model_dependency=RandomForestRegressor,
    ):
        return super()._set_model(
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
            model_dependency=model_dependency,
        )
