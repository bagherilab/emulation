import unittest
from unittest.mock import patch

import pandas as pd

from permutation.loader import CSVLoader


class CSVLoaderTests(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]
        )

    @patch("permutation.loader.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.test_df
        test_features = ["a", "b"]
        test_response = "c"
        test_test_size = 0.3
        test_seed = 100

        testLoader = CSVLoader(
            path="test",
            features=test_features,
            response=test_response,
            test_size=test_test_size,
            seed=test_seed,
        )

        expected_X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        expected_y = pd.Series([3, 6, 9], name="c")

        self.assertTrue(testLoader._X.equals(expected_X))
        self.assertTrue(testLoader._y.equals(expected_y))
