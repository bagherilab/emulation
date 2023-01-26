import unittest
from unittest.mock import Mock, MagicMock, patch

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from permutation.models.svr import SVReg
from permutation.models.hyperparameters import HParams


class SVRegTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sklearn_SVReg_mock = MagicMock(spec=SVR)
        self.sklearn_scalar_mock = MagicMock(spec=StandardScaler)
        self.expected_pipeline = Pipeline(
            [("_", self.sklearn_scalar_mock), ("_", self.sklearn_SVReg_mock)]
        )
        self.X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        self.y = pd.Series([3, 6, 9], name="c")

        self.hparams_mock = Mock(spec=HParams)
        self.hparams_mock.as_dict = Mock(return_value={"C": 1.0, "epsilon": 2})
        self.hparams_mock.__str__ = Mock(return_value="{'C': 1.0, 'epsilon': 2}")

    def test_svr_initialization_without_hparams(self) -> None:
        testSVReg = SVReg.set_model(
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )

        self.assertEqual(
            testSVReg.algorithm_name,
            "Support Vector Regression",
        )
        self.assertEqual(testSVReg.algorithm_type, "Regression")
        self.assertIs(testSVReg.hparams, None)
        self.sklearn_SVReg_mock.assert_called_once_with()
        self.sklearn_scalar_mock.assert_called_once()
        self.assertEqual(len(testSVReg.pipeline), 2)

    def test_svr_initialization_with_hparams(self) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )

        self.assertEqual(
            testSVReg.algorithm_name,
            "Support Vector Regression, hparams: {'C': 1.0, 'epsilon': 2}",
        )
        self.assertEqual(testSVReg.algorithm_type, "Regression")
        self.assertIsNot(testSVReg.hparams, None)
        self.sklearn_SVReg_mock.assert_called_once_with(C=1.0, epsilon=2)
        self.sklearn_scalar_mock.assert_called_once()
        self.assertEqual(len(testSVReg.pipeline), 2)

    def test_svr_initialization_with_preprocessing(self) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )

        self.sklearn_SVReg_mock.assert_called_once_with(C=1.0, epsilon=2)

    def test_svr_initialization_no_preprocessing(self) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=None,
            model_dependency=self.sklearn_SVReg_mock,
        )

        self.sklearn_SVReg_mock.assert_called_once_with(C=1.0, epsilon=2)
        self.assertEqual(len(testSVReg.pipeline), 1)

    def test_fit_model_calls_fit_and_returns_fit_pesvrormance(self) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )
        testSVReg.pipeline.fit = Mock()
        predicted_df = self.y + 2  # force RMSE to be 2
        testSVReg.pipeline.predict = Mock(return_value=predicted_df)

        returned_value = testSVReg.fit_model(self.X, self.y)

        testSVReg.pipeline.fit.assert_called_once_with(self.X, self.y)
        testSVReg.pipeline.predict.assert_called_once_with(self.X)
        self.assertEqual(returned_value, 2.0)

    def test_fit_model_calls_fit_and_returns_fit_pesvrormance(self) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )
        testSVReg.pipeline.fit = Mock()
        predicted_df = self.y + 2  # force RMSE to be 2
        testSVReg.pipeline.predict = Mock(return_value=predicted_df)

        returned_value = testSVReg.fit_model(self.X, self.y)

        testSVReg.pipeline.fit.assert_called_once_with(self.X, self.y)
        testSVReg.pipeline.predict.assert_called_once_with(self.X)
        self.assertEqual(returned_value, 2.0)

    @patch("permutation.models.sklearnmodel.cross_val_score")
    def test_fit_model_calls_fit_and_returns_fit_pesvrormance(self, cross_val_score_mock) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )

        return_list = [5.0] * 10
        cross_val_score_mock.return_value = np.array(return_list)
        returned_value = testSVReg.crossval_hparams(self.X, self.y, 10).values

        cross_val_score_mock.assert_called_once_with(testSVReg.pipeline, self.X, self.y, cv=10)
        self.assertEqual(returned_value, return_list)

    @patch("permutation.models.sklearnmodel.permutation_importance")
    def test_model_calls_permutation_and_returns_permutation_scores(self, permutation_mock) -> None:
        testSVReg = SVReg.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_SVReg_mock,
        )

        n_repeats = 30
        mock_output = Mock()
        mock_output.importances = MagicMock()
        mock_output.importances.__getitem__.return_value = np.zeros((1, n_repeats))
        permutation_mock.return_value = mock_output

        return_value = testSVReg.permutation(self.X, self.y, repeats=n_repeats)

        inspected_names = [metric.name for metric in return_value]
        inspected_values = [metric.values for metric in return_value]
        expected_names = ["Feature:a", "Feature:b"]
        expected_values = [[0.0] * n_repeats] * 2

        # two features
        self.assertEqual(mock_output.importances.__getitem__.call_count, 2)
        self.assertEqual(inspected_names, expected_names)
        self.assertEqual(inspected_values, expected_values)
