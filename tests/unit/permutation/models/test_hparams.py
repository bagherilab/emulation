import unittest

from permutation.models.hyperparameters import Hparams


class BatchMetricTests(unittest.TestCase):
    def test_empty_defaults(self) -> None:
        testHparams = Hparams()

        self.assertFalse(testHparams.values)
        self.assertFalse(testHparams.args)
        self.assertEqual(testHparams.number_of_parameters, 0)

    def test_update(self) -> None:
        testHparams = Hparams()

        testHparams.update_params("alpha", 1.0)

        self.assertEqual(testHparams.args, ["alpha"])
        self.assertEqual(testHparams.values, [1.0])
        self.assertEqual(testHparams.number_of_parameters, 1)

    def test_multiple_updates(self) -> None:
        testHparams = Hparams()

        testHparams.update_params("alpha", 1.0)
        testHparams.update_params("beta", "foo")

        self.assertEqual(testHparams.args, ["alpha", "beta"])
        self.assertEqual(testHparams.values, [1.0, "foo"])
        self.assertEqual(testHparams.number_of_parameters, 2)

    def test_mutability(self) -> None:
        _ = Hparams()
        _.update_params("alpha", 1.0)

        testHparams = Hparams()

        self.assertFalse(testHparams.values)
        self.assertFalse(testHparams.args)
        self.assertEqual(testHparams.number_of_parameters, 0)
