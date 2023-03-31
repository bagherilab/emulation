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

    def test_split_data(self, mock_read_csv):
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
                "col2": [5, 6, 7, 8],
                "col3": [9, 10, np.inf, 12],
                "col4": [13, 14, 15, 16],
                "y": [1, 2, np.nan, 4],
            }
        )
        mock_read_csv.return_value = X

        testLoader = CSVLoader(
            path="test",
            features=["col1", "col2", "col3", "col4"],
            response=["y"],
            test_size=self.test_size,
            seed=self.seed,
        )

        expected_removed_cols = ["col1", "col3"]
        expected_removed_rows = pd.DataFrame(
            {
                "col2": [7],
                "col4": [15],
                "y": [np.nan],
            },
            index=[2],
        )

        expected_cleaned = pd.DataFrame(
            {
                "col2": [5, 6, 8],
                "col4": [13, 14, 16],
                "y": [1.0, 2.0, 4.0],
            },
            index=[0, 1, 2],
        )

        removed_cols, removed_rows = testLoader.clean_data()
        cleaned_X, cleaned_y = testLoader.load_working_data()
        cleaned_data = pd.concat([cleaned_X, cleaned_y], axis=1)

        self.assertListEqual(removed_cols.values.tolist(), expected_removed_cols)
        self.assertTrue(removed_rows.equals(expected_removed_rows))
        self.assertTrue(cleaned_data.equals(expected_cleaned))
