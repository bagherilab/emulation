import unittest

from permutation.models.hyperparameters import Hparams


class HparametersTest(unittest.TestCase):
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

    def test_cv_metrics(self) -> None:
        testHparams = Hparams()
        testHparams.update_params("alpha", 1.0)

        test_metrics = [10.0, 11.0, 12.0]
        testHparams.update_cv_metrics(test_metrics)

        self.assertEqual(testHparams.cross_validation_performance.values, test_metrics)
        self.assertEqual(testHparams.performance_average, 11.0)

    def test_block_update_if_cv_performed(self) -> None:
        def improper_order():
            testHparams = Hparams()
            testHparams.update_params("alpha", 1.0)

            test_metrics = [10.0, 11.0, 12.0]
            testHparams.update_cv_metrics(test_metrics)

            testHparams.update_params("beta", 2.0)

        self.assertRaises(AttributeError, improper_order)

    def test_as_dict(self) -> None:
        testHparams = Hparams()

        testHparams.update_params("alpha", 1.0)
        testHparams.update_params("beta", "foo")

        expected = {"alpha": 1.0, "beta": "foo"}

        self.assertEqual(testHparams.as_dict(), expected)
