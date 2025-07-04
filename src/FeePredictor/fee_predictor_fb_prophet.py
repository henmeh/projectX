import pandas as pd
from sqlalchemy import create_engine, text, Table, MetaData, Column, DateTime, Numeric, String, insert, inspect
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import datetime as dt
import numpy as np
import logging
import pickle
import os
import itertools # Used implicitly by ParameterGrid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Prophet parameters (as a starting point/defaults) ---
DEFAULT_PROPHET_PARAMETERS = {
    'growth': 'linear',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'daily_seasonality': True,
    'weekly_seasonality': True,
    'yearly_seasonality': False, 
    'seasonality_mode': 'additive' # Added default for seasonality_mode
}

# --- Search space for hyperparameter tuning ---
PROPHET_PARAM_GRID = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.2],
    'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0],
    'daily_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'yearly_seasonality': [True, False],
    'seasonality_mode': ['additive', 'multiplicative'] 
}

# A simplified grid for "very_short" lookbacks where extensive seasonality might not be beneficial
PROPHET_PARAM_GRID_VERY_SHORT = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1],
    'seasonality_prior_scale': [0.1, 1.0, 5.0],
    'daily_seasonality': [True], # Often daily is most relevant for very short term
    'weekly_seasonality': [False], # Not enough data for weekly in 3 days
    'yearly_seasonality': [False], # Not enough data for yearly in 3 days
    'seasonality_mode': ['additive'] 
}

# Minimum data points to enable certain seasonalities in Prophet
# Prophet needs at least 2 full cycles for a seasonality to be meaningful.
# Hourly data, so 24 points/day.
MIN_DAILY_DATA_POINTS = 2 * 24 
MIN_WEEKLY_DATA_POINTS = 2 * 24 * 7 
MIN_YEARLY_DATA_POINTS = 2 * 24 * 365 

class FeePredictorProphet:
    """
    A class to fetch fee data from PostgreSQL, train models using Facebook Prophet,
    and predict future fees while enforcing logical ordering.
    """
    
    def __init__(self, db_connection_string, historical_table_name, 
                 prediction_table_name='fee_predictions_prophet',
                 lookback_intervals=None, forecast_horizon_hours=24,
                 model_dir='./trained_models_prophet/',
                 tuning_enabled=True,
                 tuning_metrics=['mape'], 
                 tuning_cv_initial_days=60, # Initial training period for cross-validation
                 tuning_cv_period_days=15,   # Spacing between cutoff points for cross-validation
                 tuning_param_grid=None 
                ):
        self.db_connection_string = db_connection_string
        self.historical_table_name = historical_table_name
        self.prediction_table_name = prediction_table_name
        self.forecast_horizon_hours = forecast_horizon_hours
        self.model_dir = model_dir
        self.generated_at = dt.datetime.now()
        
        if lookback_intervals is None:
            self.lookback_intervals = {
                "short_term": "3h",
                "medium_term": "3d",
                "long_term": "3w"
            }
        else:
            # Ensure keys are lowercase for consistent access and file naming
            self.lookback_intervals = {k.lower(): v.lower() for k, v in lookback_intervals.items()}
            
        self.engine = create_engine(self.db_connection_string)
        self.metadata = MetaData()

        # Define the prediction table schema
        self.fee_predictions_table = Table(
            self.prediction_table_name, self.metadata,
            Column('prediction_time', DateTime, nullable=False, primary_key=True), # Added primary_key to prediction_time
            Column('model_name', String(50), nullable=False, primary_key=True), # Added primary_key to model_name for composite key
            Column('fast_fee', Numeric, nullable=False),
            Column('medium_fee', Numeric, nullable=False),
            Column('low_fee', Numeric, nullable=False),
            Column('generated_at', DateTime, nullable=False),
            # You might want a UniqueConstraint here if you only want one prediction per model_name at a given time
            # UniqueConstraint('prediction_time', 'model_name', name='uq_prediction_model_time')
        )

        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trained_models_by_lookback = {}
        self.latest_predictions = {}
        
        self.tuning_enabled = tuning_enabled
        self.tuning_metrics = tuning_metrics
        # Convert days to timedelta strings for Prophet's cross_validation
        self.tuning_cv_initial = f"{tuning_cv_initial_days} days"
        self.tuning_cv_period = f"{tuning_cv_period_days} days"
        self.tuning_cv_horizon = f"{self.forecast_horizon_hours}h"
        self.tuning_param_grid = tuning_param_grid if tuning_param_grid is not None else PROPHET_PARAM_GRID
        
        self._create_tables() 
        logging.info("FeePredictor initialized and database tables checked.")

    def _create_tables(self):
        """Creates necessary database tables if they don't exist."""
        try:
            self.metadata.create_all(self.engine) # Creates all tables defined in self.metadata
            logging.info(f"Table '{self.prediction_table_name}' checked/created successfully.")

            with self.engine.connect() as conn:
                # Use sqlalchemy.inspect() for deprecation warning fix
                insp = inspect(conn)
                if not insp.has_table(self.historical_table_name):
                    logging.critical(f"Historical data table '{self.historical_table_name}' does not exist in the database. Please ensure it is created and populated.")
                    raise RuntimeError(f"Required historical data table '{self.historical_table_name}' not found.")
        except Exception as e:
            logging.critical(f"Failed to create/check database tables: {e}")
            raise 

    def _fetch_fee_data(self):
        """Fetches fee data from the PostgreSQL database."""
        try:
            # Use text() for arbitrary SQL queries in SQLAlchemy 2.0 style
            query = text(f"SELECT timestamp, fast_fee, medium_fee, low_fee FROM {self.historical_table_name} ORDER BY timestamp")
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn) # Pass connection object directly
            if df.empty:
                logging.warning(f"No data fetched from PostgreSQL table '{self.historical_table_name}'. It's empty.")
                return None
            logging.info(f"Data fetched successfully from PostgreSQL table '{self.historical_table_name}'. Rows: {len(df)}")
            return df
        except Exception as e:
            logging.error(f"Error fetching data from PostgreSQL database: {e}")
            logging.error("Please ensure your DB_CONNECTION_STRING, historical_table_name, and column names are correct and the table has data.")
            return None

    @staticmethod
    def _create_features(df):
        """
        Creates time-based features from a datetime index.
        (Primarily for models like RandomForest; Prophet handles these internally from 'ds').
        This method is kept for consistency but not strictly necessary for Prophet.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            # Ensure the index is a DatetimeIndex before creating features
            df.index = pd.to_datetime(df.index)
            
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        return df

    def _preprocess_data(self, df):
        """Prepares the data for model training."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # _create_features is called here, but Prophet primarily uses the 'ds' column directly.
        # It doesn't harm to have these, but they are not used by Prophet itself.
        df_with_features = self._create_features(df) 
        
        target_fees = ['fast_fee', 'medium_fee', 'low_fee']
        for col in target_fees:
            if col not in df_with_features.columns:
                logging.error(f"Missing expected fee column: {col}")
                raise ValueError(f"Missing required column '{col}' in historical data.")
            # Convert to numeric, coercing errors to NaN. Then fill NaNs if desired (e.g., with 0 or last valid value)
            df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce')
            # Important: Decide how to handle NaNs for Prophet. Prophet usually handles NaNs internally by interpolating or treating as missing data,
            # but it's often better to explicitly deal with them for targets.
            # For simplicity, let's just drop rows where all fee values are NaN in this example.
            # If a specific fee_type is all NaN, it will be skipped during training.

        initial_rows = len(df_with_features)
        # Drop rows where ALL target fee columns are NaN
        df_with_features.dropna(subset=target_fees, how='all', inplace=True)
        if len(df_with_features) < initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df_with_features)} rows due to all fee values being NaN after preprocessing.")
            
        # Features are not strictly needed for Prophet, but keeping for consistency if other models were added.
        # X_all represents the index (ds) for Prophet.
        X_all = df_with_features.index.to_frame(name='ds') 
        y_all_dict = {fee: df_with_features[fee] for fee in target_fees} 
        
        logging.info(f"Data preprocessed. Total observations after cleaning: {len(X_all)}")
        return X_all, y_all_dict

    def _get_min_data_points(self, params):
        """
        Calculate minimum data points required based on enabled seasonalities and forecast horizon.
        Prophet generally needs at least two cycles of a seasonality to detect it.
        """
        min_required = 0
        if params.get('daily_seasonality'):
            min_required = max(min_required, MIN_DAILY_DATA_POINTS) # 2 days of data
        if params.get('weekly_seasonality'):
            min_required = max(min_required, MIN_WEEKLY_DATA_POINTS) # 2 weeks of data
        if params.get('yearly_seasonality'):
            min_required = max(min_required, MIN_YEARLY_DATA_POINTS) # 2 years of data
        
        # Prophet also needs data extending beyond the history point to make predictions.
        # Ensure there's at least enough data for the forecast horizon, plus some baseline.
        # A common recommendation is at least 3-4x the horizon duration for initial fit.
        min_required = max(min_required, self.forecast_horizon_hours * 3, 30) # At least 30 points as a general minimum
        return min_required

    def _find_best_prophet_params(self, df_prophet, model_key_name, metric_to_optimize='mape'):
        """
        Finds the best Prophet parameters using cross-validation.
        
        Args:
            df_prophet (pd.DataFrame): DataFrame with 'ds' and 'y' columns for Prophet training.
            model_key_name (str): Name of the model being tuned (e.g., 'hourly').
            metric_to_optimize (str): The metric to minimize ('rmse', 'mae', 'mape', etc.).
        
        Returns:
            dict: The best parameters found.
        """
        logging.info(f"    Starting hyperparameter tuning for {model_key_name} using metric '{metric_to_optimize}'.")
        
        # Use a simpler grid for very short lookbacks if explicitly defined
        param_grid_to_use = PROPHET_PARAM_GRID_VERY_SHORT if model_key_name == "very_short" else self.tuning_param_grid
        
        all_params = list(ParameterGrid(param_grid_to_use))
        
        # Ensure default parameters are always considered as a baseline
        if DEFAULT_PROPHET_PARAMETERS not in all_params:
            all_params.insert(0, DEFAULT_PROPHET_PARAMETERS.copy()) # Use .copy() to avoid modifying the original default

        best_params = DEFAULT_PROPHET_PARAMETERS.copy() 
        best_metric_value = float('inf') 

        logging.info(f"    Testing {len(all_params)} parameter combinations...")

        # Calculate the minimum duration of data needed for cross-validation
        # initial + horizon + period for at least one cutoff
        min_cv_data_duration = pd.Timedelta(self.tuning_cv_initial) + \
                               pd.Timedelta(self.tuning_cv_horizon) + \
                               pd.Timedelta(self.tuning_cv_period)

        actual_data_duration = df_prophet['ds'].max() - df_prophet['ds'].min() if not df_prophet.empty else pd.Timedelta(0)
        
        if df_prophet.empty or actual_data_duration < min_cv_data_duration:
            logging.warning(f"    Not enough historical data ({actual_data_duration} duration) for meaningful cross-validation with initial={self.tuning_cv_initial}, period={self.tuning_cv_period}, horizon={self.tuning_cv_horizon}. Skipping tuning. Using default Prophet parameters.")
            return DEFAULT_PROPHET_PARAMETERS.copy()

        for i, params in enumerate(all_params):
            try:
                # Ensure all required parameters are present, using defaults if missing from the grid
                current_params = DEFAULT_PROPHET_PARAMETERS.copy()
                current_params.update(params)

                # Skip combinations that don't make sense for the current data size
                min_data_for_these_params = self._get_min_data_points(current_params)
                if len(df_prophet) < min_data_for_these_params:
                    logging.debug(f"      Skipping combo {i+1}/{len(all_params)} ({current_params}) due to insufficient data ({len(df_prophet)} points, need {min_data_for_these_params}) for its enabled seasonalities.")
                    continue

                model = Prophet(
                    growth=current_params['growth'],
                    changepoint_prior_scale=current_params['changepoint_prior_scale'],
                    seasonality_prior_scale=current_params['seasonality_prior_scale'],
                    seasonality_mode=current_params['seasonality_mode'],
                    daily_seasonality=current_params['daily_seasonality'],
                    weekly_seasonality=current_params['weekly_seasonality'],
                    yearly_seasonality=current_params['yearly_seasonality']
                )
                
                # Fit the model: remove suppress_stdout_stderror as it's not a direct argument to .fit()
                # Prophet typically logs via its own logger, which can be configured externally.
                model.fit(df_prophet) 
                
                # Perform cross-validation
                df_cv = cross_validation(
                    model, 
                    initial=self.tuning_cv_initial, 
                    period=self.tuning_cv_period, 
                    horizon=self.tuning_cv_horizon,
                    # parallel="processes" is good for production but can be removed for debugging if issues arise
                    parallel="processes" 
                )
                
                if df_cv.empty:
                    logging.debug(f"      Cross-validation returned empty DataFrame for parameters: {current_params}. Skipping.")
                    continue

                df_p = performance_metrics(df_cv, metrics=self.tuning_metrics)
                
                if df_p.empty or metric_to_optimize not in df_p.columns:
                    logging.debug(f"      Performance metrics empty or missing '{metric_to_optimize}' for parameters: {current_params}. Skipping.")
                    continue

                current_metric_value = df_p[metric_to_optimize].mean()

                logging.info(f"      Combo {i+1}/{len(all_params)} ({current_params}) - Avg {metric_to_optimize}: {current_metric_value:.4f}")

                if current_metric_value < best_metric_value:
                    best_metric_value = current_metric_value
                    best_params = current_params # Store the exact parameters used
                    logging.info(f"      New best parameters found: {best_params} with {metric_to_optimize}: {best_metric_value:.4f}")

            except Exception as e:
                logging.error(f"      Error during tuning for parameters {params}: {e}")
                logging.exception("Detailed traceback for tuning error:")
                continue

        logging.info(f"    Tuning completed for {model_key_name}. Best parameters: {best_params} with avg {metric_to_optimize}: {best_metric_value:.4f}")
        return best_params

    def _train_models(self, X_data_ds_only, y_data_dict, lookback_timedelta=None, model_key_name=None, tuned_params=None):
        """
        Trains a separate Facebook Prophet model for each fee category,
        filtering data by a lookback timedelta, optionally with tuned parameters.
        Returns a tuple of (trained_models_dict, parameters_used_for_training).
        """
        if lookback_timedelta:
            # X_data_ds_only is a DataFrame with a 'ds' column (which is also the index)
            end_time = X_data_ds_only['ds'].max()
            start_time = end_time - lookback_timedelta
            
            # Filter X_data_ds_only based on 'ds' column values
            X_train_filtered_ds = X_data_ds_only[X_data_ds_only['ds'] >= start_time]
            
            # Filter y_data_dict based on the index (timestamps)
            y_train_dict_filtered = {fee: y_data_dict[fee][y_data_dict[fee].index >= start_time] 
                                     for fee in y_data_dict.keys()}
            
            logging.info(f"  Training with lookback: {str(lookback_timedelta)} (Data points: {len(X_train_filtered_ds)})")
        else:
            X_train_filtered_ds = X_data_ds_only 
            y_train_dict_filtered = y_data_dict
            logging.info(f"  Training with ALL available data (Data points: {len(X_train_filtered_ds)})")

        # Check if there's any data after filtering
        if not any(not series.empty for series in y_train_dict_filtered.values()):
            logging.warning("  No data available for the specified lookback window for ANY fee type. Skipping model training for this lookback.")
            return {}, {} # Return empty dicts

        trained_models = {}
        
        # Use tuned parameters if provided, otherwise fall back to default
        params_to_use = tuned_params if tuned_params is not None else DEFAULT_PROPHET_PARAMETERS.copy()

        for fee_type, y_train_series in y_train_dict_filtered.items():
            if y_train_series.empty:
                logging.warning(f"    No target data for {fee_type} in this lookback. Skipping training.")
                continue

            y_train_clean = y_train_series.dropna() 
            
            if y_train_clean.empty:
                logging.warning(f"    {fee_type} has no valid non-NaN data points after cleaning. Skipping training.")
                continue

            # Create the Prophet-specific DataFrame with 'ds' and 'y'
            # Ensure 'ds' is timezone-naive if your DB timestamps are naive
            prophet_df = pd.DataFrame({'ds': y_train_clean.index, 'y': y_train_clean.values})
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None) if prophet_df['ds'].dt.tz is not None else prophet_df['ds']

            min_required_data = self._get_min_data_points(params_to_use)

            if len(prophet_df) < min_required_data:
                logging.warning(f"    Insufficient data ({len(prophet_df)} points, need at least {min_required_data}) for {fee_type} for meaningful Prophet training with enabled seasonalities. Skipping.")
                continue

            logging.info(f"    Training Prophet model for {fee_type} with {len(prophet_df)} data points...")
            logging.info(f"    Using parameters for {fee_type}: {params_to_use}")
            
            try:
                model = Prophet(
                    growth=params_to_use.get('growth', 'linear'), 
                    changepoint_prior_scale=params_to_use['changepoint_prior_scale'],
                    seasonality_prior_scale=params_to_use['seasonality_prior_scale'],
                    seasonality_mode=params_to_use.get('seasonality_mode', 'additive'), 
                    daily_seasonality=params_to_use['daily_seasonality'],
                    weekly_seasonality=params_to_use['weekly_seasonality'],
                    yearly_seasonality=params_to_use['yearly_seasonality']
                )
                
                # Add any custom seasonalities or holidays here if needed (e.g., model.add_country_holidays(country_name='US'))
                
                # Fit the model - removed suppress_stdout_stderror
                model.fit(prophet_df) 
                trained_models[fee_type] = model
                logging.info(f"    Prophet model for {fee_type} trained successfully.")
            except Exception as e:
                logging.error(f"    Error training Prophet model for {fee_type}: {e}")
                logging.exception("Detailed traceback for Prophet training error:")
                if fee_type in trained_models:
                    del trained_models[fee_type]

        # Return the trained models and the parameters that were actually used
        return trained_models, params_to_use

    def _save_models(self, models, parameters_used, model_name_prefix):
        """Saves trained Prophet models and their associated parameters to disk."""
        # It's generally safer to save models individually if they are large or if you expect to load them piecemeal.
        # However, for simplicity and since they are small, saving as a bundle works.
        saved_info = {
            'models': models,
            'parameters_used': parameters_used
        }
        filename = os.path.join(self.model_dir, f"{model_name_prefix}_all.pkl")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(saved_info, f)
            logging.info(f"  Saved all models and their parameters for '{model_name_prefix}' as {filename}")
        except Exception as e:
            logging.error(f"  Failed to save model bundle {filename}: {e}")
            logging.exception("Detailed traceback for model saving error:")

    def _load_models(self, model_name_prefix):
        """Loads trained Prophet models and their parameters from disk."""
        filename = os.path.join(self.model_dir, f"{model_name_prefix}_all.pkl")
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    saved_info = pickle.load(f)
                    loaded_models = saved_info.get('models', {})
                    loaded_params_used = saved_info.get('parameters_used', {}) 
                    
                    expected_fees = ['fast_fee', 'medium_fee', 'low_fee']
                    # Verify that all expected models are present and are indeed Prophet objects
                    if not all(fee in loaded_models and isinstance(loaded_models[fee], Prophet) for fee in expected_fees):
                        logging.warning(f"  Incomplete or incorrect model set found in {filename}. Forcing retraining.")
                        return None, None # Return None for both models and params
                    
                    logging.info(f"  Loaded models and parameters for '{model_name_prefix}' from {filename}")
                    return loaded_models, loaded_params_used # Return both
            except Exception as e:
                logging.warning(f"  Could not load model bundle {filename}: {e}. Will force retraining.")
                logging.exception("Detailed traceback for model loading error:")
                return None, None
        else:
            logging.info(f"  No saved model bundle found for '{model_name_prefix}' at {filename}. Will train new models.")
            return None, None

    def _predict_future_fees(self, trained_models, current_time):
        """
        Predicts future fees using Prophet models and enforces logical ordering:
        low_fee <= medium_fee <= fast_fee.
        """
        if not trained_models:
            logging.error("No trained models provided for prediction for this lookback. Cannot predict.")
            return pd.DataFrame()

        # Start prediction from the next full hour after current_time
        start_prediction_time = (current_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)).replace(tzinfo=None) # Ensure timezone-naive
        
        target_future_timestamps = pd.date_range(start=start_prediction_time, 
                                                 periods=self.forecast_horizon_hours, 
                                                 freq='h')
        
        future_prophet_df = pd.DataFrame({'ds': target_future_timestamps})
        # Ensure the 'ds' column in the future DataFrame is timezone-naive if Prophet expects it.
        # Prophet handles timezones internally, but input 'ds' usually should be naive or consistent.
        future_prophet_df['ds'] = future_prophet_df['ds'].dt.tz_localize(None) if future_prophet_df['ds'].dt.tz is not None else future_prophet_df['ds']
        
        if future_prophet_df.empty:
            logging.warning("Target future timestamps DataFrame is empty based on forecast_horizon_hours. Cannot predict.")
            return pd.DataFrame()
        
        # Initialize combined predictions DataFrame with the target timestamps as index
        predictions_df_combined = pd.DataFrame(index=target_future_timestamps)

        for fee_type, model in trained_models.items():
            if model:
                try:
                    forecast = model.predict(future_prophet_df)
                    # Align forecast 'yhat' with the target timestamps
                    aligned_forecast_series = forecast.set_index('ds')['yhat'].reindex(target_future_timestamps)
                    predictions_df_combined[fee_type] = aligned_forecast_series
                except Exception as e:
                    logging.error(f"Error during prediction for {fee_type}: {e}")
                    logging.exception("Detailed traceback for prediction error:")
                    predictions_df_combined[fee_type] = np.nan 
            else:
                logging.warning(f"  Model for {fee_type} was not found or was None. Skipping prediction for this fee type.")
                predictions_df_combined[fee_type] = np.nan
                
        # Enforce logical ordering and non-negativity
        corrected_predictions_df = predictions_df_combined.copy()

        for index, row in predictions_df_combined.iterrows():
            if row.isnull().any(): # If any fee is NaN, mark all as NaN for consistency
                corrected_predictions_df.loc[index, ['low_fee', 'medium_fee', 'fast_fee']] = np.nan 
                continue 
                
            low = row.get('low_fee', 0) # Default to 0 for sorting if somehow missing (though checked above)
            medium = row.get('medium_fee', 0)
            fast = row.get('fast_fee', 0)

            # Ensure values are numeric before sorting
            if pd.notna(low) and pd.notna(medium) and pd.notna(fast):
                sorted_fees = sorted([low, medium, fast])
                
                # Assign sorted values and ensure non-negativity
                corrected_predictions_df.loc[index, 'low_fee'] = max(0, sorted_fees[0])
                corrected_predictions_df.loc[index, 'medium_fee'] = max(0, sorted_fees[1])
                corrected_predictions_df.loc[index, 'fast_fee'] = max(0, sorted_fees[2])
            else:
                # If any input was NaN, ensure output is NaN
                corrected_predictions_df.loc[index, ['low_fee', 'medium_fee', 'fast_fee']] = np.nan
                
        return corrected_predictions_df

    def _store_predictions_to_db(self, predictions_df, model_name):
        """Stores the generated predictions into the database table 'fee_predictions'."""
        if predictions_df.empty:
            logging.warning(f"No predictions DataFrame provided to store for model '{model_name}'.")
            return

        records_to_insert = []

        for index, row in predictions_df.iterrows():
            # Only insert if all fee types are valid (not NaN)
            if not row[['fast_fee', 'medium_fee', 'low_fee']].isnull().any(): 
                records_to_insert.append({
                    'prediction_time': index.to_pydatetime(), # Convert Timestamp to Python datetime
                    'model_name': model_name,
                    'fast_fee': float(row['fast_fee']), # Ensure numeric types for DB
                    'medium_fee': float(row['medium_fee']),
                    'low_fee': float(row['low_fee']),
                    'generated_at': self.generated_at
                })
        
        if records_to_insert:
            try:
                with self.engine.connect() as conn:
                    # Delete existing predictions for this model and prediction times
                    # This ensures idempotence and prevents primary key violations on subsequent runs
                    #delete_stmt = self.fee_predictions_table.delete().where(
                    #    self.fee_predictions_table.c.model_name == model_name,
                    #    self.fee_predictions_table.c.prediction_time.in_([rec['prediction_time'] for rec in records_to_insert])
                    #)
                    #conn.execute(delete_stmt)
                    
                    conn.execute(insert(self.fee_predictions_table), records_to_insert)
                    conn.commit() 
                logging.info(f"Successfully stored {len(records_to_insert)} predictions for model '{model_name}' to '{self.prediction_table_name}'.")
            except Exception as e:
                logging.error(f"Error storing predictions for model '{model_name}': {e}")
                logging.exception("Detailed traceback for DB storage error:")
        else:
            logging.warning(f"No valid non-NaN predictions to store for model '{model_name}'.")

    def run(self, retrain_interval_hours=24):
        """
        Executes the full prediction pipeline: fetches data, trains/loads models for various
        lookback periods, predicts future fees, and stores the results.
        """
        logging.info(f"Starting FeePredictor run at {dt.datetime.now()}...")
        
        current_time_for_prediction = pd.to_datetime(dt.datetime.now())

        df_all = self._fetch_fee_data()
        if df_all is None or df_all.empty:
            logging.error("No historical data available. Cannot proceed with prediction.")
            self.latest_predictions = {}
            return self.latest_predictions

        # X_all here is a DataFrame containing only the 'ds' column (which is the timestamp index)
        # y_all_dict contains series of fee values indexed by timestamp
        X_all, y_all_dict = self._preprocess_data(df_all)
        
        if len(X_all) < 100: # General check, more specific checks happen per model
            logging.error(f"Insufficient total historical data ({len(X_all)} points) to perform meaningful training and prediction. Need at least 100 points. Please populate '{self.historical_table_name}'.")
            self.latest_predictions = {}
            return self.latest_predictions

        # X_train_full is effectively just the 'ds' (timestamp) column from the preprocessed data
        X_train_full_ds = X_all 
        y_train_full_dict = y_all_dict 

        logging.info(f"Total historical data observations: {len(X_train_full_ds)}")
        logging.info(f"Data available for lookback slicing based on index range: {X_train_full_ds['ds'].min()} to {X_train_full_ds['ds'].max()}")

        self.latest_predictions = {} 
        self.trained_models_by_lookback = {} 

        # Sort lookbacks from shortest to longest. This might be useful for some dependencies or logging clarity.
        sorted_lookbacks = sorted(self.lookback_intervals.items(), key=lambda item: pd.Timedelta(item[1]))

        for name, interval_str in sorted_lookbacks:
            logging.info(f"\n--- Processing {name.replace('_', ' ').title()} Model ({interval_str} Lookback) ---")
            
            try:
                lookback_timedelta = pd.Timedelta(interval_str)
            except ValueError as e:
                logging.error(f"  Invalid lookback interval string '{interval_str}' for model '{name}': {e}. Please use valid pandas Timedelta strings (e.g., '3h', '30d', '1w'). Skipping this model.")
                continue 
            
            model_name_prefix = f"model_{name}"
            trained_models_for_interval = None
            params_used_for_interval = None # This will store the parameters that were actually used for training

            last_training_time_path = os.path.join(self.model_dir, f"{model_name_prefix}_last_trained.txt")
            retrain_needed = True
            
            if os.path.exists(last_training_time_path):
                try:
                    with open(last_training_time_path, 'r') as f:
                        last_trained_str = f.read().strip()
                        last_trained_time = dt.datetime.fromisoformat(last_trained_str)
                        time_since_last_train = current_time_for_prediction - last_trained_time
                        if time_since_last_train < dt.timedelta(hours=retrain_interval_hours):
                            logging.info(f"  Models for {name} trained {time_since_last_train} ago, less than {retrain_interval_hours} hours. Attempting to load existing models.")
                            loaded_models, loaded_params = self._load_models(model_name_prefix)
                            if loaded_models and loaded_params:
                                trained_models_for_interval = loaded_models
                                params_used_for_interval = loaded_params
                                logging.info(f"  Loaded models and parameters: {params_used_for_interval}")
                                retrain_needed = False
                            else:
                                logging.warning(f"  Failed to load models for {name} (e.g., file missing or corrupted). Retraining required.")
                        else:
                            logging.info(f"  Models for {name} trained {time_since_last_train} ago, more than {retrain_interval_hours} hours. Retraining.")

                except Exception as e:
                    logging.warning(f"  Could not read last trained time for {name}: {e}. Retraining required.")
            else:
                logging.info(f"  No previous training record found for {name}. Retraining required.")

            if retrain_needed:
                logging.info(f"  Retraining models for {name} lookback...")
                
                # Filter data for tuning/training based on the lookback timedelta
                end_time_for_lookback = X_train_full_ds['ds'].max()
                start_time_for_lookback = end_time_for_lookback - lookback_timedelta
                
                # For tuning, we need a Prophet-compatible DataFrame for at least one fee type
                # The data used for tuning must be within the specified lookback window
                y_data_for_lookback_dict_for_tuning = {
                    fee: y_train_full_dict[fee][y_train_full_dict[fee].index >= start_time_for_lookback].dropna()
                    for fee in y_train_full_dict.keys()
                }
                
                tune_fee_type = None
                if 'fast_fee' in y_data_for_lookback_dict_for_tuning and not y_data_for_lookback_dict_for_tuning['fast_fee'].empty:
                    tune_fee_type = 'fast_fee'
                else: # Fallback to any non-empty fee type if fast_fee is not available
                    for f_type, f_series in y_data_for_lookback_dict_for_tuning.items():
                        if not f_series.empty:
                            tune_fee_type = f_type
                            break

                best_tuned_params = DEFAULT_PROPHET_PARAMETERS.copy() # Initialize with defaults
                
                if self.tuning_enabled and tune_fee_type:
                    logging.info(f"  Initiating hyperparameter tuning for {name} model using {tune_fee_type} data...")
                    
                    prophet_df_for_tuning = pd.DataFrame({
                        'ds': y_data_for_lookback_dict_for_tuning[tune_fee_type].index,
                        'y': y_data_for_lookback_dict_for_tuning[tune_fee_type].values
                    })
                    prophet_df_for_tuning['ds'] = prophet_df_for_tuning['ds'].dt.tz_localize(None) if prophet_df_for_tuning['ds'].dt.tz is not None else prophet_df_for_tuning['ds']

                    # Check for sufficient data *duration* for cross-validation
                    min_tuning_data_duration = pd.Timedelta(self.tuning_cv_initial) + \
                                               pd.Timedelta(self.tuning_cv_horizon) + \
                                               pd.Timedelta(self.tuning_cv_period)
                    
                    actual_data_duration_for_tuning = prophet_df_for_tuning['ds'].max() - prophet_df_for_tuning['ds'].min() if not prophet_df_for_tuning.empty else pd.Timedelta(0)

                    if prophet_df_for_tuning.empty or actual_data_duration_for_tuning < min_tuning_data_duration:
                         logging.warning(f"  Not enough data duration ({actual_data_duration_for_tuning}) for tuning {name} model (need at least {min_tuning_data_duration}). Skipping tuning. Using default Prophet parameters.")
                         # best_tuned_params remains DEFAULT_PROPHET_PARAMETERS.copy()
                    else:
                        best_tuned_params = self._find_best_prophet_params(
                            prophet_df_for_tuning, 
                            model_key_name=name, 
                            metric_to_optimize=self.tuning_metrics[0]
                        )
                        if not best_tuned_params:
                            logging.warning(f"  Tuning failed or returned no best parameters for {name}. Using default Prophet parameters.")
                            best_tuned_params = DEFAULT_PROPHET_PARAMETERS.copy()
                else:
                    if not self.tuning_enabled:
                        logging.info(f"  Hyperparameter tuning is disabled. Using default Prophet parameters for {name}.")
                    else: # tuning_enabled is True but tune_fee_type is None
                        logging.warning(f"  No valid fee data found to perform tuning for {name}. Using default Prophet parameters.")
                    # best_tuned_params remains DEFAULT_PROPHET_PARAMETERS.copy()

                # Train Models with (potentially) tuned parameters
                # Pass X_train_full_ds (which is a DataFrame of 'ds' values) and y_train_full_dict (series of y values)
                trained_models_for_interval, params_used_for_interval = self._train_models(
                    X_train_full_ds, # This is just for getting the overall time range
                    y_train_full_dict, 
                    lookback_timedelta, 
                    model_key_name=name, 
                    tuned_params=best_tuned_params
                )
                
                if trained_models_for_interval: 
                    self._save_models(trained_models_for_interval, params_used_for_interval, model_name_prefix) # Pass the parameters explicitly
                    try:
                        with open(last_training_time_path, 'w') as f:
                            f.write(current_time_for_prediction.isoformat())
                    except Exception as e:
                        logging.error(f"  Failed to write last trained time for {name}: {e}")
                else:
                    logging.warning(f"  No models trained successfully for {name} lookback. Skipping prediction for this lookback.")
                    continue 

            if not trained_models_for_interval:
                logging.warning(f"  No trained models available for {name} after load/train attempt. Skipping prediction and storage.")
                continue

            self.trained_models_by_lookback[name] = trained_models_for_interval 
            
            logging.info(f"  Attempting to predict future fees for {name} model...")
            current_predictions = self._predict_future_fees(trained_models_for_interval, current_time_for_prediction)
            
            if current_predictions.empty:
                logging.warning(f"  No predictions generated for {name} (DataFrame was empty after prediction). Skipping storage for this lookback.")
                continue

            self.latest_predictions[name] = current_predictions

            self._store_predictions_to_db(current_predictions, name)
        
        if not self.latest_predictions:
            logging.info("FeePredictor run completed. No predictions were successfully generated and stored across all models.")
        else:
            logging.info("FeePredictor run completed. Predictions stored in DB and memory.")
            
        return self.latest_predictions

if __name__ == "__main__":
    # IMPORTANT: Replace with your actual database connection string, user, password, host, port, and database name
    # IMPORTANT: Replace with your actual PostgreSQL credentials
    DB_CONN_STR = 'postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain'
    HISTORICAL_FEE_TABLE = 'mempool_fee_histogram'
    PREDICTION_TABLE = 'fee_predictions_prophet' # Make sure this table exists and matches the schema below

    custom_lookbacks = {
        "very_short": "3d",  
        "hourly": "3w",      
        "daily": "180d",     
        "weekly": "365d"     
    }

    try:
        predictor = FeePredictorProphet(
            db_connection_string=DB_CONN_STR,
            historical_table_name=HISTORICAL_FEE_TABLE,
            prediction_table_name=PREDICTION_TABLE,
            lookback_intervals=custom_lookbacks, 
            forecast_horizon_hours=48, # Predict 48 hours into the future
            model_dir='./trained_models_prophet/',
            tuning_enabled=True, 
            tuning_metrics=['mape'], # Mean Absolute Percentage Error
            tuning_cv_initial_days=60, # Initial training period for CV
            tuning_cv_period_days=15,  # Spacing between cutoff points
        )

        predictions_result = predictor.run(retrain_interval_hours=24) 

        if predictions_result:
            logging.info("\n--- Final Predictions Summary (from current run) ---")
            for model_name, predictions_df in predictions_result.items():
                logging.info(f"\nPredictions from {model_name.replace('_', ' ').title()} Model (first 5 rows):")
                print(predictions_df.head())
                logging.info(f"... (total {len(predictions_df)} predictions for {model_name})")
                logging.info("-" * 50)
        else:
            logging.info("No predictions successfully generated and stored.")
            
    except RuntimeError as e:
        logging.critical(f"Application terminated due to critical setup error: {e}")
    except Exception as e:
        logging.critical(f"An unhandled error occurred during execution: {e}")
        logging.exception("Unhandled exception traceback:")