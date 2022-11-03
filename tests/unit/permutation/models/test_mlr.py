import unittest
from unittest.mock import Mock, MagicMock, patch

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from permutation.models.mlr import MLR
from permutation.models.hyperparameters import HParams


class MLRTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sklearn_EN_mock = MagicMock(spec=ElasticNet)
        self.sklearn_scalar_mock = MagicMock(spec=StandardScaler)
        self.expected_pipeline = Pipeline(
            [("scalar_mock", self.sklearn_scalar_mock), ("mlr", self.sklearn_EN_mock)]
        )
        self.X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        self.y = pd.Series([3, 6, 9], name="c")

        self.hparams_mock = Mock(spec=HParams)
        self.hparams_mock.as_dict = Mock(return_value={"alpha": 1, "l1_ratio": 0.5})
        self.hparams_mock.__str__ = Mock(return_value="{'alpha': 1, 'l1_ratio': 0.5}")

    def test_mlr_initialization_without_hparams(self) -> None:
        testMLR = MLR.set_model(
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )

        self.assertEqual(
            testMLR.algorithm_name,
            "Regularized Linear Regression",
        )
        self.assertEqual(testMLR.algorithm_type, "Regression")
        self.assertIs(testMLR.hparams, None)
        self.sklearn_EN_mock.assert_called_once_with()
        self.sklearn_scalar_mock.assert_called_once()
        self.assertEqual(len(testMLR.pipeline), 2)

    def test_mlr_initialization_with_hparams(self) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )

        self.assertEqual(
            testMLR.algorithm_name,
            "Regularized Linear Regression, hparams: {'alpha': 1, 'l1_ratio': 0.5}",
        )
        self.assertEqual(testMLR.algorithm_type, "Regression")
        self.assertIsNot(testMLR.hparams, None)
        self.sklearn_EN_mock.assert_called_once_with(alpha=1, l1_ratio=0.5)
        self.sklearn_scalar_mock.assert_called_once()
        self.assertEqual(len(testMLR.pipeline), 2)

    def test_mlr_initialization_with_preprocessing(self) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )

        self.sklearn_EN_mock.assert_called_once_with(alpha=1, l1_ratio=0.5)

    def test_mlr_initialization_no_preprocessing(self) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=None,
            model_dependency=self.sklearn_EN_mock,
        )

        self.sklearn_EN_mock.assert_called_once_with(alpha=1, l1_ratio=0.5)
        self.assertEqual(len(testMLR.pipeline), 1)

    def test_fit_model_calls_fit_and_returns_fit_performance(self) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )
        testMLR.pipeline.fit = Mock()
        predicted_df = self.y + 2  # force RMSE to be 2
        testMLR.pipeline.predict = Mock(return_value=predicted_df)

        returned_value = testMLR.fit_model(self.X, self.y)

        testMLR.pipeline.fit.assert_called_once_with(self.X, self.y)
        testMLR.pipeline.predict.assert_called_once_with(self.X)
        self.assertEqual(returned_value, 2.0)

    def test_fit_model_calls_fit_and_returns_fit_performance(self) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )
        testMLR.pipeline.fit = Mock()
        predicted_df = self.y + 2  # force RMSE to be 2
        testMLR.pipeline.predict = Mock(return_value=predicted_df)

        returned_value = testMLR.fit_model(self.X, self.y)

        testMLR.pipeline.fit.assert_called_once_with(self.X, self.y)
        testMLR.pipeline.predict.assert_called_once_with(self.X)
        self.assertEqual(returned_value, 2.0)

    @patch("permutation.models.sklearnmodel.cross_val_score")
    def test_fit_model_calls_fit_and_returns_fit_performance(self, cross_val_score_mock) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )

        return_list = [5.0] * 10
        cross_val_score_mock.return_value = np.array(return_list)
        returned_value = testMLR.crossval_hparams(self.X, self.y, 10).values

        cross_val_score_mock.assert_called_once_with(testMLR.pipeline, self.X, self.y, cv=10)
        self.assertEqual(returned_value, return_list)

    @patch("permutation.models.sklearnmodel.permutation_importance")
    def test_model_calls_permutation_and_returns_permutation_scores(self, permutation_mock) -> None:
        testMLR = MLR.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_EN_mock,
        )

        n_repeats = 30
        mock_output = Mock()
        mock_output.importances = MagicMock()
        mock_output.importances.__getitem__.return_value = np.zeros((1, n_repeats))
        permutation_mock.return_value = mock_output

        return_value = testMLR.permutation(self.X, self.y, repeats=n_repeats)

        inspected_names = [metric.name for metric in return_value]
        inspected_values = [metric.values for metric in return_value]
        expected_names = ["Feature: a", "Feature: b"]
        expected_values = [[0.0] * n_repeats] * 2

        # two features
        self.assertEqual(mock_output.importances.__getitem__.call_count, 2)
        self.assertEqual(inspected_names, expected_names)
        self.assertEqual(inspected_values, expected_values)
