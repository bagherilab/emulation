# type: ignore
from typing import List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class StratifiedKFolder(BaseCrossValidator):
    """Stratified K-Folds cross-validator"""

    def __init__(
        self,
        stratify: str,
        n_splits: int = 10,
        shuffle: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[list[Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        layouts, counts = self._get_layout_counts(X)
        if self.shuffle:
            rng = self.random_state if self.random_state is not None else np.random.default_rng()
            rng.shuffle(layouts)

        folds: list[list] = [[] for _ in range(self.n_splits)]
        for layout, count in zip(layouts, counts):
            fold_sizes = np.array([count // self.n_splits] * self.n_splits)
            fold_sizes[: count % self.n_splits] += 1
            if self.shuffle:
                rng.shuffle(fold_sizes)

            indices = np.flatnonzero(X[self.stratify] == layout)
            current = 0
            for fold, fold_size in zip(folds, fold_sizes):
                fold.extend(indices[current : current + fold_size])
                current += fold_size

        for fold in folds:
            training_indices = np.array(list(set(range(len(X))) - set(fold)))
            test_indices = np.array(fold)
            yield training_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[list[Any]] = None,
    ) -> int:
        return super().get_n_splits(X, y, groups)

    def _get_layout_counts(self, X: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        layouts, counts = np.unique(X[self.stratify].values, return_counts=True)
        return layouts, counts
