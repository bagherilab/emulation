import unittest
from unittest.mock import Mock, MagicMock, patch

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from permutation.models.rf import RF
from permutation.models.hyperparameters import HParams


class RFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sklearn_RF_mock = MagicMock(spec=RandomForestRegressor)
        self.sklearn_scalar_mock = MagicMock(spec=StandardScaler)
        self.expected_pipeline = Pipeline(
            [("scalar_mock", self.sklearn_scalar_mock), ("rf", self.sklearn_RF_mock)]
        )
        self.X = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=["a", "b"])
        self.y = pd.Series([3, 6, 9], name="c")

        self.hparams_mock = Mock(spec=HParams)
        self.hparams_mock.as_dict = Mock(return_value={"n_estimators": 200, "max_features": "auto"})
        self.hparams_mock.__str__ = Mock(return_value="{'n_estimators': 1, 'max_features': 'auto'}")

    def test_rf_initialization_without_hparams(self) -> None:
        testRF = RF.set_model(
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )

        self.assertEqual(
            testRF.algorithm_name,
            "Random Forest",
        )
        self.assertEqual(testRF.algorithm_type, "Regression")
        self.assertIs(testRF.hparams, None)
        self.sklearn_RF_mock.assert_called_once_with()
        self.sklearn_scalar_mock.assert_called_once()
        self.assertEqual(len(testRF.pipeline), 2)

    def test_rf_initialization_with_hparams(self) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )

        self.assertEqual(
            testRF.algorithm_name,
            "Random Forest, hparams: {'n_estimators': 1, 'max_features': 'auto'}",
        )
        self.assertEqual(testRF.algorithm_type, "Regression")
        self.assertIsNot(testRF.hparams, None)
        self.sklearn_RF_mock.assert_called_once_with(n_estimators=200, max_features="auto")
        self.sklearn_scalar_mock.assert_called_once()
        self.assertEqual(len(testRF.pipeline), 2)

    def test_rf_initialization_with_preprocessing(self) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )

        self.sklearn_RF_mock.assert_called_once_with(n_estimators=200, max_features="auto")

    def test_rf_initialization_no_preprocessing(self) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=None,
            model_dependency=self.sklearn_RF_mock,
        )

        self.sklearn_RF_mock.assert_called_once_with(n_estimators=200, max_features="auto")
        self.assertEqual(len(testRF.pipeline), 1)

    def test_fit_model_calls_fit_and_returns_fit_performance(self) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )
        testRF.pipeline.fit = Mock()
        predicted_df = self.y + 2  # force RMSE to be 2
        testRF.pipeline.predict = Mock(return_value=predicted_df)

        returned_value = testRF.fit_model(self.X, self.y)

        testRF.pipeline.fit.assert_called_once_with(self.X, self.y)
        testRF.pipeline.predict.assert_called_once_with(self.X)
        self.assertEqual(returned_value, 2.0)

    def test_fit_model_calls_fit_and_returns_fit_performance(self) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )
        testRF.pipeline.fit = Mock()
        predicted_df = self.y + 2  # force RMSE to be 2
        testRF.pipeline.predict = Mock(return_value=predicted_df)

        returned_value = testRF.fit_model(self.X, self.y)

        testRF.pipeline.fit.assert_called_once_with(self.X, self.y)
        testRF.pipeline.predict.assert_called_once_with(self.X)
        self.assertEqual(returned_value, 2.0)

    @patch("permutation.models.sklearnmodel.cross_val_score")
    def test_fit_model_calls_fit_and_returns_fit_performance(self, cross_val_score_mock) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )

        return_list = [5.0] * 10
        cross_val_score_mock.return_value = np.array(return_list)
        returned_value = testRF.crossval_hparams(self.X, self.y, 10).values

        cross_val_score_mock.assert_called_once_with(testRF.pipeline, self.X, self.y, cv=10)
        self.assertEqual(returned_value, return_list)

    @patch("permutation.models.sklearnmodel.permutation_importance")
    def test_model_calls_permutation_and_returns_permutation_scores(self, permutation_mock) -> None:
        testRF = RF.set_model(
            hparams=self.hparams_mock,
            preprocessing_dependencies=[("scalar_mock", self.sklearn_scalar_mock)],
            model_dependency=self.sklearn_RF_mock,
        )

        n_repeats = 30
        mock_output = Mock()
        mock_output.importances = MagicMock()
        mock_output.importances.__getitem__.return_value = np.zeros((1, n_repeats))
        permutation_mock.return_value = mock_output

        return_value = testRF.permutation(self.X, self.y, repeats=n_repeats)

        inspected_names = [metric.name for metric in return_value]
        inspected_values = [metric.values for metric in return_value]
        expected_names = ["Feature:a", "Feature:b"]
        expected_values = [[0.0] * n_repeats] * 2

        # two features
        self.assertEqual(mock_output.importances.__getitem__.call_count, 2)
        self.assertEqual(inspected_names, expected_names)
        self.assertEqual(inspected_values, expected_values)
