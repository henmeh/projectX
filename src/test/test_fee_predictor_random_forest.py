import unittest
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import tempfile
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from FeePredictor.fee_predictor_random_forest import FeePredictorRandomForest  # Replace with your actual module name

class TestFeePredictorRandomForest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = self.temp_dir.name
        self.mock_db_conn = 'sqlite:///:memory:'
        self.predictor = FeePredictorRandomForest(
            self.mock_db_conn, 
            'historical', 
            model_dir=self.model_dir
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def mock_data(self):
        dates = pd.date_range(start='2025-08-01', periods=100, freq='H', tz='UTC')
        df = pd.DataFrame({
            'timestamp': dates,
            'fast_fee': np.random.uniform(5, 20, 100),
            'medium_fee': np.random.uniform(3, 15, 100),
            'low_fee': np.random.uniform(1, 10, 100)
        })
        return df

    @patch('pandas.read_sql')
    def test_fetch_fee_data(self, mock_read_sql):
        mock_df = self.mock_data()
        mock_read_sql.return_value = mock_df
        df = self.predictor._fetch_fee_data()
        self.assertFalse(df.empty)
        self.assertEqual(df['timestamp'].dt.tz.zone, 'UTC')

    def test_preprocess_data(self):
        mock_df = self.mock_data()
        X, y_dict = self.predictor._preprocess_data(mock_df)
        self.assertIn('hour', X.columns)
        self.assertEqual(len(y_dict), 3)
        self.assertEqual(X.index.tz.zone, 'UTC')

    def test_train_models(self):
        mock_df = self.mock_data()
        X, y_dict = self.predictor._preprocess_data(mock_df)
        models = self.predictor._train_models(X, y_dict, model_key_name='very_short')
        self.assertEqual(len(models), 3)
        for model in models.values():
            self.assertIsInstance(model, RandomForestRegressor)

    def test_predict_future_fees(self):
        mock_df = self.mock_data()
        X, y_dict = self.predictor._preprocess_data(mock_df)
        models = self.predictor._train_models(X, y_dict, model_key_name='very_short')
        current_time = datetime.now(timezone.utc)
        predictions = self.predictor._predict_future_fees(models, current_time)
        self.assertEqual(len(predictions), self.predictor.forecast_horizon_hours)
        self.assertTrue((predictions['low_fee'] <= predictions['medium_fee']).all())
        self.assertTrue((predictions['medium_fee'] <= predictions['fast_fee']).all())
        self.assertEqual(predictions.index.tz.zone, 'UTC')

    @patch('pandas.read_sql')
    def test_load_latest_predictions(self, mock_read_sql):
        mock_df = pd.DataFrame({
            'prediction_time': pd.date_range(start=datetime.now(timezone.utc), periods=24, freq='H'),
            'model_name': ['very_short'] * 24,
            'fast_fee': [10.0] * 24,
            'medium_fee': [5.0] * 24,
            'low_fee': [2.0] * 24,
            'generated_at': [datetime.now(timezone.utc)] * 24
        })
        mock_read_sql.return_value = mock_df
        loaded = self.predictor.load_latest_predictions(freshness_hours=1)
        self.assertTrue(loaded)
        self.assertIn('very_short', self.predictor.latest_predictions)
        self.assertEqual(self.predictor.latest_predictions['very_short'].index.tz.zone, None)  # Since read_sql may strip tz; adjust if needed

    @patch('pandas.read_sql')
    def test_tune_hyperparameters(self, mock_read_sql):
        mock_df = self.mock_data()
        mock_read_sql.return_value = mock_df
        best_params = self.predictor.tune_random_forest_hyperparameters(n_iter_search=2, cv_folds=2)
        self.assertIn('very_short', best_params)
        self.assertIn('fast_fee', best_params['very_short'])

if __name__ == '__main__':
    unittest.main()