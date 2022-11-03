import unittest
from unittest.mock import Mock
from typing import Any, Iterable

import config_utils.assign_hyperparameters as MOD


def check_values_in_range(minimum: Any, maximum: Any, iterable: Iterable) -> bool:
    return any([minimum <= val <= maximum for val in iterable])


class SequentialMetricTests(unittest.TestCase):
    def test_build(self) -> None:
        # User writes config files in YAML files
        self.config = Mock()
        self.config.hparams = Mock()

        continuous_params = {
            "alpha": {"type": "int", "range": [1, 10]},
            "beta": {"type": "float", "range": [0.0, 1.0]},
        }
        discrete_params = {"foo": ["a", "b", "c"], "bar": ["x", "y", "z"], "bool": [True, False]}
        static_params = {"static": "test"}

        self.config.hparams.configure_mock(
            continuous=continuous_params, discrete=discrete_params, static=static_params
        )

        # User uses the config object to make a Dataframe of values
        resolved_df = MOD.build_hparams_df(self.config.hparams)
        print(resolved_df)
        # Returned df should have attributes of original config YAML with pseudo-random numbers
        self.assertEqual(resolved_df.shape, (288, 6))
        self.assertTrue(check_values_in_range(1, 10, resolved_df.alpha))
        self.assertTrue(check_values_in_range(0.0, 1.0, resolved_df.beta))
        self.assertEqual(resolved_df["foo"].unique().tolist(), ["a", "b", "c"])
        self.assertEqual(resolved_df["bar"].unique().tolist(), ["x", "y", "z"])
        self.assertEqual(resolved_df["bool"].unique().tolist(), [True, False])
