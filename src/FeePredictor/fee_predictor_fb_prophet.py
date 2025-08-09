import pandas as pd
from sqlalchemy import create_engine, text, Table, MetaData, Column, DateTime, Numeric, String, inspect
from sqlalchemy.dialects.postgresql import insert as pg_insert
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import datetime as dt
import numpy as np
import logging
import pickle
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
FEE_TYPES = ['low_fee', 'medium_fee', 'fast_fee']

# Default parameters provide a sensible baseline.
DEFAULT_PROPHET_PARAMETERS = {
    'growth': 'linear',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'daily_seasonality': True,
    'weekly_seasonality': True,
    'yearly_seasonality': False,
    'seasonality_mode': 'additive'
}

# Expanded parameter grid for more thorough hyperparameter tuning.
PROPHET_PARAM_GRID = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
    'seasonality_prior_scale': [1.0, 5.0, 10.0, 20.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

class FeePredictorProphet:
    """
    Fetches fee data, trains Prophet models with robust timezone handling,
    and predicts future fees while enforcing logical ordering.
    """
    def __init__(self, db_connection_string, historical_table_name,
                 prediction_table_name='fee_predictions_prophet',
                 lookback_intervals=None, forecast_horizon_hours=24,
                 model_dir='./trained_models_prophet/',
                 tuning_enabled=True):
        
        self.db_connection_string = db_connection_string
        self.historical_table_name = historical_table_name
        self.prediction_table_name = prediction_table_name
        self.forecast_horizon_hours = forecast_horizon_hours
        self.model_dir = model_dir
        self.tuning_enabled = tuning_enabled
        
        self.lookback_intervals = lookback_intervals or {
            "short_term": "3d",
            "medium_term": "2w",
            "long_term": "8w"
        }
        
        self.engine = create_engine(self.db_connection_string)
        self.metadata = MetaData()

        # Define the prediction table with a composite primary key and an ON CONFLICT UPDATE clause for PostgreSQL.
        self.fee_predictions_table = Table(
            self.prediction_table_name, self.metadata,
            Column('prediction_time', DateTime(timezone=True), primary_key=True),
            Column('model_name', String(50), primary_key=True),
            Column('fast_fee', Numeric, nullable=False),
            Column('medium_fee', Numeric, nullable=False),
            Column('low_fee', Numeric, nullable=False),
            Column('generated_at', DateTime(timezone=True), nullable=False)
        )

        os.makedirs(self.model_dir, exist_ok=True)
        self._create_tables()
        logging.info("FeePredictor initialized and database tables checked.")

    def _create_tables(self):
        """Creates the prediction table if it doesn't exist."""
        try:
            self.metadata.create_all(self.engine)
            logging.info(f"Table '{self.prediction_table_name}' checked/created successfully.")
        except Exception as e:
            logging.critical(f"Failed to create/check database tables: {e}")
            raise

    def _fetch_fee_data(self):
        """Fetches fee data, assuming timestamps in the DB are in UTC."""
        try:
            query = text(f"SELECT timestamp, {', '.join(FEE_TYPES)} FROM {self.historical_table_name} ORDER BY timestamp")
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
            
            # CRITICAL: Assume DB stores naive timestamps in UTC. Localize them immediately.
            # If your DB column is `timestamp with time zone`, this might not be necessary,
            # but it's safer to be explicit.
            if df.index.tz is None:
                df = df.tz_localize('UTC')
            else:
                df = df.tz_convert('UTC')

            logging.info(f"Fetched {len(df)} rows from '{self.historical_table_name}'.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data from PostgreSQL: {e}")
            return pd.DataFrame()

    def _preprocess_data(self, df):
        """Prepares the data for model training."""
        # Convert all fee columns to numeric, coercing errors to NaN
        for col in FEE_TYPES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where ALL target fees are NaN
        initial_rows = len(df)
        df.dropna(subset=FEE_TYPES, how='all', inplace=True)
        if len(df) < initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df)} rows where all fee values were NaN.")
        
        # Resample to hourly frequency to ensure a consistent time series, and forward-fill missing values.
        # This is crucial for Prophet to handle missing data points gracefully.
        df_resampled = df.resample('h').mean()
        df_resampled[FEE_TYPES] = df_resampled[FEE_TYPES].interpolate(method='time')
        
        logging.info(f"Data preprocessed. Observations after resampling and interpolation: {len(df_resampled)}")
        return df_resampled

    def _find_best_prophet_params(self, df_prophet, fee_type):
        """Finds the best Prophet parameters using cross-validation."""
        logging.info(f"    Starting hyperparameter tuning for {fee_type}.")
        
        param_grid = PROPHET_PARAM_GRID
        all_params = list(ParameterGrid(param_grid))
        results = []

        # Prophet needs at least 2 full cycles for seasonality. 2 weeks is a safe minimum for daily+weekly.
        if len(df_prophet) < 24 * 14:
             logging.warning("    Not enough data for cross-validation. Skipping tuning and using default parameters.")
             return DEFAULT_PROPHET_PARAMETERS

        for params in all_params:
            try:
                # Combine grid params with defaults
                current_params = {**DEFAULT_PROPHET_PARAMETERS, **params}
                m = Prophet(**current_params).fit(df_prophet)
                
                # Perform cross-validation
                df_cv = cross_validation(m, initial='14 days', period='7 days', horizon='3 days', parallel="processes", disable_tqdm=True)
                df_p = performance_metrics(df_cv, metrics=['mape'])
                
                if not df_p.empty:
                    results.append({'params': current_params, 'mape': df_p['mape'].mean()})
            except Exception as e:
                logging.error(f"      Error during tuning for params {params}: {e}")
        
        if not results:
            logging.warning("    Tuning produced no valid results. Using default parameters.")
            return DEFAULT_PROPHET_PARAMETERS

        best_params = min(results, key=lambda x: x['mape'])
        logging.info(f"    Tuning complete for {fee_type}. Best MAPE: {best_params['mape']:.4f}")
        return best_params['params']

    def _train_or_load_model(self, data, fee_type, lookback_key):
        """Trains a new model or loads an existing one from disk."""
        model_path = os.path.join(self.model_dir, f"{lookback_key}_{fee_type}_model.pkl")
        
        # Simple check: if a model file exists, load it. For production, you'd add versioning/retraining logic.
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    logging.info(f"  Loading existing model for {fee_type} ({lookback_key}) from {model_path}")
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"  Could not load model {model_path}: {e}. Retraining.")

        # Prepare data for Prophet
        df_prophet = data[[fee_type]].dropna().reset_index()
        df_prophet.rename(columns={'index': 'ds', fee_type: 'y'}, inplace=True)

        if len(df_prophet) < 50: # Need a minimum amount of data to train
            logging.warning(f"  Insufficient data ({len(df_prophet)} points) for {fee_type} ({lookback_key}). Cannot train model.")
            return None

        # Find best parameters if tuning is enabled
        params = self._find_best_prophet_params(df_prophet, fee_type) if self.tuning_enabled else DEFAULT_PROPHET_PARAMETERS
        
        logging.info(f"  Training new model for {fee_type} ({lookback_key}) with params: {params}")
        model = Prophet(**params).fit(df_prophet)
        
        # Save the newly trained model
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logging.error(f"  Failed to save model {model_path}: {e}")
            
        return model

    def _predict_and_enforce_order(self, models, current_time):
        """Predicts future fees and ensures low_fee <= medium_fee <= fast_fee."""
        # Create future dataframe, anchored to the current time and in UTC.
        future_df = models['low_fee'].make_future_dataframe(periods=self.forecast_horizon_hours, freq='h', include_history=False)
        
        predictions = pd.DataFrame(index=future_df['ds'])
        for fee_type, model in models.items():
            if model:
                forecast = model.predict(future_df)
                predictions[fee_type] = forecast['yhat'].values
        
        # Enforce logical ordering and non-negativity in a vectorized way
        predictions[FEE_TYPES] = predictions[FEE_TYPES].clip(lower=0)
        sorted_fees = np.sort(predictions[FEE_TYPES].values, axis=1)
        predictions[FEE_TYPES] = sorted_fees
        
        return predictions

    def _store_predictions_to_db(self, predictions_df, model_name):
        """Stores predictions using PostgreSQL's ON CONFLICT DO UPDATE for an upsert operation."""
        if predictions_df.empty:
            return

        records_to_insert = []
        generated_at_utc = dt.datetime.now(dt.timezone.utc)
        for timestamp, row in predictions_df.iterrows():
            if not row.isnull().any():
                records_to_insert.append({
                    'prediction_time': timestamp.to_pydatetime(),
                    'model_name': model_name,
                    'fast_fee': row['fast_fee'],
                    'medium_fee': row['medium_fee'],
                    'low_fee': row['low_fee'],
                    'generated_at': generated_at_utc
                })
        
        if not records_to_insert:
            logging.warning(f"No valid predictions to store for model '{model_name}'.")
            return

        try:
            # Use PostgreSQL's specific INSERT...ON CONFLICT statement for an efficient "upsert"
            stmt = pg_insert(self.fee_predictions_table).values(records_to_insert)
            update_dict = {c.name: c for c in stmt.excluded if c.name not in ['prediction_time', 'model_name']}
            stmt = stmt.on_conflict_do_update(
                index_elements=['prediction_time', 'model_name'],
                set_=update_dict
            )
            with self.engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
            logging.info(f"Successfully stored/updated {len(records_to_insert)} predictions for model '{model_name}'.")
        except Exception as e:
            logging.error(f"Error storing predictions for model '{model_name}': {e}")

    def run(self):
        """Executes the full prediction pipeline for all lookback intervals."""
        logging.info("--- Starting FeePredictor Run ---")
        
        full_df = self._fetch_fee_data()
        if full_df.empty:
            logging.error("No historical data available. Aborting run.")
            return

        processed_df = self._preprocess_data(full_df)
        
        current_time_utc = dt.datetime.now(dt.timezone.utc)

        for lookback_key, lookback_str in self.lookback_intervals.items():
            logging.info(f"--- Processing lookback: {lookback_key} ({lookback_str}) ---")
            
            lookback_delta = pd.to_timedelta(lookback_str)
            start_time = current_time_utc - lookback_delta
            
            lookback_data = processed_df[processed_df.index >= start_time]

            if len(lookback_data) < 100:
                logging.warning(f"Not enough data ({len(lookback_data)} points) for lookback '{lookback_key}'. Skipping.")
                continue

            trained_models = {}
            for fee in FEE_TYPES:
                trained_models[fee] = self._train_or_load_model(lookback_data, fee, lookback_key)

            # Ensure all models were trained/loaded successfully before predicting
            if all(trained_models.values()):
                predictions = self._predict_and_enforce_order(trained_models, current_time_utc)
                self._store_predictions_to_db(predictions, lookback_key)
            else:
                logging.error(f"Could not train/load all models for lookback '{lookback_key}'. Skipping prediction and storage.")

        logging.info("--- FeePredictor Run Finished ---")
