import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from permutation.loader import CSVLoader


@patch("permutation.loader.pd.read_csv")
class CSVLoaderTests(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
        self.features = ["a", "b"]
        self.response = "c"
        self.test_size = 0.3
        self.seed = 100

    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        testLoader = CSVLoader(
            path="test",
            features=self.features,
            response=self.response,
            test_size=self.test_size,
            seed=self.seed,
        )

        expected_X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        expected_y = pd.Series([3, 6, 9], name="c")

        self.assertTrue(testLoader._X.equals(expected_X))
        self.assertTrue(testLoader._y.equals(expected_y))

    def test_split_data_no_stratify(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        testLoader = CSVLoader(
            path="test",
            features=self.features,
            response=self.response,
            test_size=self.test_size,
            seed=self.seed,
        )

        X_training, y_training = testLoader.load_training_data()
        X_testing, y_testing = testLoader.load_testing_data()
        X_working, y_working = testLoader.load_working_data()

        self.assertEqual(X_training.shape, (2, 2))
        self.assertEqual(y_training.shape, (2,))
        self.assertEqual(X_testing.shape, (1, 2))
        self.assertEqual(y_testing.shape, (1,))
        self.assertEqual(X_working.shape, (3, 2))
        self.assertEqual(y_working.shape, (3,))

    def test_split_data_stratify(self, mock_read_csv):
        n_samples = 1000
        stratify_column = "Layouts"
        classes = ["A", "B", "C", "D"]
        stratified_df = pd.DataFrame(
            {
                "Feature 1": np.random.randn(n_samples),
                "Feature 2": np.random.randn(n_samples),
                stratify_column: np.random.choice(classes, size=n_samples, p=[0.5, 0.1, 0.2, 0.2]),
            }
        )

        mock_read_csv.return_value = stratified_df

        testLoader = CSVLoader(
            path="test",
            features=["Feature 1", stratify_column],
            response=["Feature 2"],
            test_size=self.test_size,
            seed=self.seed,
            stratify=stratify_column,
        )

        X_training, _ = testLoader.load_training_data()
        X_testing, _ = testLoader.load_testing_data()
        X_working, _ = testLoader.load_working_data()

        # Calculate the class distribution in the original data
        class_distribution = X_working[stratify_column].value_counts(normalize=True)

        # Calculate the class distribution in the training data
        train_class_distribution = X_training[stratify_column].value_counts(normalize=True)

        # Calculate the class distribution in the test data
        test_class_distribution = X_testing[stratify_column].value_counts(normalize=True)

        # Check that the class distribution in the training data is close to the original
        for class_name in classes:
            assert (
                abs(train_class_distribution[class_name] - class_distribution[class_name]) < 0.05
            )  # tolerance of 5%

        # Check that the class distribution in the test data is close to the original
        for class_name in class_distribution.index:
            assert abs(test_class_distribution[class_name] - class_distribution[class_name]) < 0.05

    def test_subsample(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df
        expected_X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        expected_y = pd.Series([3, 6, 9], name="c")

        testLoader = CSVLoader(
            path="test",
            features=self.features,
            response=self.response,
            test_size=0.5,
            seed=self.seed,
        )
        testLoader.subsample(2)
        X_training, y_training = testLoader.load_training_data()
        X_testing, y_testing = testLoader.load_testing_data()
        X_working, y_working = testLoader.load_working_data()
        X, y = testLoader.load_original_data()

        self.assertEqual(X_training.shape, (1, 2))
        self.assertEqual(y_training.shape, (1,))
        self.assertEqual(X_testing.shape, (1, 2))
        self.assertEqual(y_testing.shape, (1,))
        self.assertEqual(X_working.shape, (2, 2))
        self.assertEqual(y_working.shape, (2,))
        self.assertTrue(X.equals(expected_X))
        self.assertTrue(y.equals(expected_y))

    def test_clean_data(self, mock_read_csv):
        X = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4],
                "col2": [5, np.nan, 7, 8],
                "col3": [9, 10, np.inf, 12],
                "col4": [13, 14, 15, 16],
                "COMPONENTS": [1, 1, 1, 1],
                "y": [1, 2, np.nan, 4],
            }
        )
        mock_read_csv.return_value = X

        testLoader = CSVLoader(
            path="test",
            features=["col1", "col2", "col3", "col4", "COMPONENTS"],
            response="y",
            test_size=self.test_size,
            seed=self.seed,
        )

        expected_removed_cols = ["col2"]
        expected_removed_rows = pd.DataFrame(
            {
                "col1": [np.nan],
                "col2": [7.0],
                "col3": [np.inf],
                "col4": [15],
                "COMPONENTS": [1],
                "y": [np.nan],
            },
            index=[2],
        )

        expected_cleaned = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 4.0],
                "col3": [9.0, 10.0, 12.0],
                "col4": [13, 14, 16],
                "COMPONENTS": [1, 1, 1],
                "y": [1.0, 2.0, 4.0],
            },
            index=[0, 1, 2],
        )

        removed_cols, removed_rows, mult_comp_rows = testLoader.clean_data()
        cleaned_X, cleaned_y = testLoader.load_working_data()
        cleaned_data = pd.concat([cleaned_X, cleaned_y], axis=1)

        self.assertListEqual(removed_cols, expected_removed_cols)
        self.assertTrue(removed_rows.equals(expected_removed_rows))
        self.assertTrue(cleaned_data.equals(expected_cleaned))
        self.assertTrue(mult_comp_rows.empty)
