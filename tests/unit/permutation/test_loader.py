import unittest
from unittest.mock import patch

import pandas as pd

from permutation.loader import CSVLoader


@patch("permutation.loader.pd.read_csv")
class CSVLoaderTests(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
        )
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

        expected_X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        expected_y = pd.Series([3, 6, 9], name="c")

        self.assertEqual(testLoader._X_train.shape, (2, 2))
        self.assertEqual(testLoader._y_train.shape, (2,))
        self.assertEqual(testLoader._X_test.shape, (1, 2))
        self.assertEqual(testLoader._y_test.shape, (1,))

    def test_original_data_integrity(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        testLoader = CSVLoader(
            path="test",
            features=self.features,
            response=self.response,
            test_size=self.test_size,
            seed=self.seed,
        )

        testLoader._X_working.loc["a", 1] = 0
        testLoader._y_working[0] = 2

        self.assertFalse(testLoader._X.equals(testLoader._X_working))
        self.assertFalse(testLoader._y.equals(testLoader._y_working))

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

        self.assertEqual(testLoader._X_train.shape, (1, 2))
        self.assertEqual(testLoader._y_train.shape, (1,))
        self.assertEqual(testLoader._X_test.shape, (1, 2))
        self.assertEqual(testLoader._y_test.shape, (1,))

    def test_unload(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df

        testLoader = CSVLoader(
            path="test",
            features=self.features,
            response=self.response,
            test_size=self.test_size,
            seed=self.seed,
        )

        testLoader.unload_data()

        self.assertIsNone(testLoader._X)
        self.assertIsNone(testLoader._y)
        self.assertIsNone(testLoader._X_working)
        self.assertIsNone(testLoader._y_working)
        self.assertIsNone(testLoader._X_train)
        self.assertIsNone(testLoader._X_test)
        self.assertIsNone(testLoader._y_train)
        self.assertIsNone(testLoader._y_test)
