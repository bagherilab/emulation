import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from permutation.models.stratified_k_fold import StratifiedKFolder


class TestStratifiedKFolder(unittest.TestCase):
    def test_stratified_k_folder(self):
        # Generate dummy data
        n_samples = 1000
        classes = ["A", "B", "C", "D"]
        stratify_column = "Layouts"
        stratified_df = pd.DataFrame(
            {
                "Feature 1": np.random.randn(n_samples),
                "Feature 2": np.random.randn(n_samples),
                stratify_column: np.random.choice(classes, size=n_samples, p=[0.5, 0.1, 0.2, 0.2]),
            }
        )
        skf = StratifiedKFolder(stratify=stratify_column, n_splits=10)

        # Ensure each fold has a similar distribution of layouts as the original data
        layout_counts = stratified_df[stratify_column].value_counts()
        for train_index, test_index in skf.split(stratified_df, stratified_df[stratify_column]):
            train_counts = stratified_df.loc[train_index, stratify_column].value_counts()
            test_counts = stratified_df.loc[test_index, stratify_column].value_counts()

            for layout in layout_counts.index:
                total_distribution = layout_counts[layout] / len(stratified_df)
                train_distribution = train_counts.get(layout, 0) / len(train_index)
                test_distribution = test_counts.get(layout, 0) / len(test_index)
                print(total_distribution, train_distribution, test_distribution)
                assert abs(train_distribution - total_distribution) < 0.05  # tolerance of 5%
                assert abs(test_distribution - total_distribution) < 0.05

        # Ensure each fold contains unique indices
        seen_indices = set()
        for train_index, test_index in skf.split(stratified_df, stratified_df[stratify_column]):
            train_set = set(train_index)
            test_set = set(test_index)
            assert len(train_set & test_set) == 0
            assert len(train_set) + len(test_set) == len(stratified_df)

        # Ensure all indices are included in the splits
        all_indices = set(range(len(stratified_df)))
        seen_indices = set()
        for train_index, test_index in skf.split(stratified_df, stratified_df[stratify_column]):
            seen_indices.update(train_index)
            seen_indices.update(test_index)
        assert seen_indices == all_indices
