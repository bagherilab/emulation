import unittest
from unittest.mock import patch

import pandas as pd

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

        self.assertEqual(X_training.shape, (1, 2))
        self.assertEqual(y_training.shape, (1,))
        self.assertEqual(X_testing.shape, (1, 2))
        self.assertEqual(y_testing.shape, (1,))
