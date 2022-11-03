import unittest

from permutation.models.hyperparameters import HParams


class HParametersTest(unittest.TestCase):
    def test_empty_defaults(self) -> None:
        testHParams = HParams()

        self.assertFalse(testHParams.values)
        self.assertFalse(testHParams.args)
        self.assertEqual(testHParams.number_of_parameters, 0)

    def test_update(self) -> None:
        testHParams = HParams()

        testHParams.update_param("alpha", 1.0)

        self.assertEqual(testHParams.args, ["alpha"])
        self.assertEqual(testHParams.values, [1.0])
        self.assertEqual(testHParams.number_of_parameters, 1)

    def test_multiple_updates(self) -> None:
        testHParams = HParams()

        testHParams.update_param("alpha", 1.0)
        testHParams.update_param("beta", "foo")

        self.assertEqual(testHParams.args, ["alpha", "beta"])
        self.assertEqual(testHParams.values, [1.0, "foo"])
        self.assertEqual(testHParams.number_of_parameters, 2)

    def test_mutability(self) -> None:
        _ = HParams()
        _.update_param("alpha", 1.0)

        testHParams = HParams()

        self.assertFalse(testHParams.values)
        self.assertFalse(testHParams.args)
        self.assertEqual(testHParams.number_of_parameters, 0)

    def test_as_dict(self) -> None:
        testHParams = HParams()

        testHParams.update_param("alpha", 1.0)
        testHParams.update_param("beta", "foo")

        expected = {"alpha": 1.0, "beta": "foo"}

        self.assertEqual(testHParams.as_dict(), expected)
