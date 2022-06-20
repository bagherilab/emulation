from abc import ABC, abstractmethod


class AbstractSKLearnModel(ABC):
    algorithm_name: str
    algorithm_type: str
    hparams: Hparams

    @classmethod
    @abstractmethod
    def set_model(
        cls, normalization=StandardScaler, model_dependency=ElasticNet
    ) -> SKLearnModel:
        ...

    def crossval_hparams(
        self, X: Any, y: Any, hparams: Hyperparams, stage_check: bool
    ) -> list[float]:
        if not stage_check:
            raise_correct_stage(Stage.VAL)

        return cross_val_score(self._pipeline, X, y, cv=cv).tolist()

    def fit_model(self, X: Any, y: Any, stage_check: bool) -> float:
        if not stage_check:
            raise_correct_stage(Stage.TRAIN)

        self._pipeline.fit(X, y)
        return self._pipeline.score(X, y)

    def performance(self, X: Any, y: Any, stage_check: bool) -> float:
        if not stage_check:
            raise_correct_stage(Stage.TEST)
        return self._pipeline.score(X, y)

    def permutation(self, X: Any, y: Any, stage_check: bool) -> list[float]:
        if not stage_check:
            raise_correct_stage(Stage.PERM)
        self._pipeline.fit(X, y)
        score, permutation_scores, pvalue = permutation_test_score(
            self.Pipeline, X, y, scoring="RMSE"
        )
        return permutation_scores.tolist()


def raise_correct_stage(stage):
    raise IncorrectStageException(stage)
