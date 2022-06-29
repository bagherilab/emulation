import unittest

import pandas as pd

from permutation.models.sklearnmodel import root_mean_square_error


class RMSETest(unittest.TestCase):
    def test_RMSE(self):
        y = pd.Series([3, 6, 9], name="c")

        for given in [2, -2, 4, -4]:
            with self.subTest(given=given):
                self.assertEqual(root_mean_square_error(y, y + given), abs(given))
