from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from permutation.models.modelprotocol import Model
from permutation.models.sklearnmodel import AbstractSKLearnModel


class RF(AbstractSKLearnModel):
    """todo"""

    algorithm_name = "Random Forest"
    algorithm_type = "Regression"

    @classmethod
    def set_model(
        cls,
        model_dependency=RandomForestRegressor,
        hparams=None,
        preprocessing_dependencies=[("scaler", StandardScaler)],
    ) -> Model:
        """todo"""

        return super()._set_model(
            model_dependency=model_dependency,
            hparams=hparams,
            preprocessing_dependencies=preprocessing_dependencies,
        )
