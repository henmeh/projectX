# fee_predictor_random_forest.py
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sqlalchemy import create_engine, text, Table, MetaData, Column, DateTime, Numeric, String, insert
from datetime import datetime, timezone, timedelta
import numpy as np
import logging
import pickle  # For saving/loading models
import os  # For managing model files
from typing import Dict, Optional, Tuple, Any

parameters = {
    "very_short": {"fast_fee": {'bootstrap': False, 'max_depth': 7, 'max_features': 1.0, 'min_samples_leaf': 2, 'min_samples_split': 13, 'n_estimators': 207},
                   "medium_fee": {'bootstrap': False, 'max_depth': 49, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 14, 'n_estimators': 90},
                   "low_fee": {'bootstrap': False, 'max_depth': 7, 'max_features': 1.0, 'min_samples_leaf': 2, 'min_samples_split': 13, 'n_estimators': 207}},

    "hourly":     {"fast_fee": {'bootstrap': True, 'max_depth': 47, 'max_features': 1.0, 'min_samples_leaf': 4, 'min_samples_split': 14, 'n_estimators': 209},
                   "medium_fee": {'bootstrap': False, 'max_depth': 6, 'max_features': 0.8, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 253},
                   "low_fee": {'bootstrap': True, 'max_depth': 27, 'max_features': 0.6, 'min_samples_leaf': 8, 'min_samples_split': 5, 'n_estimators': 153}},

    "daily":      {"fast_fee": {'bootstrap': False, 'max_depth': 32, 'max_features': 'sqrt', 'min_samples_leaf': 7, 'min_samples_split': 2, 'n_estimators': 178},
                   "medium_fee": {'bootstrap': True, 'max_depth': 33, 'max_features': 0.6, 'min_samples_leaf': 8, 'min_samples_split': 8, 'n_estimators': 171},
                   "low_fee": {'bootstrap': True, 'max_depth': 18, 'max_features': 0.6, 'min_samples_leaf': 9, 'min_samples_split': 16, 'n_estimators': 64}},
    
    "weekly":     {"fast_fee": {'bootstrap': False, 'max_depth': 5, 'max_features': 0.8, 'min_samples_leaf': 8, 'min_samples_split': 12, 'n_estimators': 292},
                   "medium_fee": {'bootstrap': False, 'max_depth': 5, 'max_features': 0.8, 'min_samples_leaf': 8, 'min_samples_split': 12, 'n_estimators': 292},
                   "low_fee": {'bootstrap': False, 'max_depth': 5, 'max_features': 0.8, 'min_samples_leaf': 8, 'min_samples_split': 12, 'n_estimators': 292}}
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeePredictorRandomForest:
    """
    A class to fetch fee data from PostgreSQL, train models with various lookback periods,
    and predict future fees while enforcing logical ordering.
    """
    
    def __init__(self, db_connection_string: str, historical_table_name: str, 
                 prediction_table_name: str = 'fee_predictions_random_forest',
                 lookback_intervals: Optional[Dict[str, str]] = None, forecast_horizon_hours: int = 24,
                 model_dir: str = './trained_models_random_forest/'):
        """
        Initializes the FeePredictor with database and prediction parameters.

        Args:
            db_connection_string (str): SQLAlchemy connection string for PostgreSQL.
            historical_table_name (str): Name of the historical fee data table.
            prediction_table_name (str): Name of the table to store predictions. Defaults to 'fee_predictions_random_forest'.
            lookback_intervals (dict, optional): Dictionary of lookback intervals for training.
                                                Keys are model names (e.g., 'very_short'),
                                                values are pandas Timedelta strings (e.g., '3H').
                                                Defaults to predefined intervals.
            forecast_horizon_hours (int): Number of hours into the future to predict. Defaults to 24.
            model_dir (str): Directory to save/load trained models. Defaults to './trained_models_random_forest/'.
        """
        self.db_connection_string = db_connection_string
        self.historical_table_name = historical_table_name
        self.prediction_table_name = prediction_table_name
        self.forecast_horizon_hours = forecast_horizon_hours
        self.model_dir = model_dir
        self.generated_at = datetime.now(timezone.utc)
        
        if lookback_intervals is None:
            self.lookback_intervals = {
                "very_short": "3H",
                "hourly": "3D",
                "daily": "3W",
                "weekly": "3M"
            }
        else:
            self.lookback_intervals = lookback_intervals
            
        self.engine = create_engine(self.db_connection_string)
        self.metadata = MetaData() # For reflecting/defining tables for core SQL operations

        # Define the prediction table structure without ORM class
        # Ensure these match your external DDL for the 'fee_predictions_random_forest' table
        self.fee_predictions_table = Table(
            self.prediction_table_name, self.metadata,
            Column('prediction_time', DateTime(timezone=True), nullable=False),
            Column('model_name', String(50), nullable=False),
            Column('fast_fee', Numeric, nullable=False),
            Column('medium_fee', Numeric, nullable=False),
            Column('low_fee', Numeric, nullable=False),
            Column('generated_at', DateTime(timezone=True), nullable=False)
        )

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trained_models_by_lookback = {} # To store trained models (loaded or newly trained)
        self.latest_predictions = {} # To store the most recent predictions
        logging.info("FeePredictorRandomForest initialized.")

    def _fetch_fee_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches fee data from the PostgreSQL database.
        Assumes the table has 'timestamp', 'fast_fee', 'medium_fee', and 'low_fee' columns.
        """
        try:
            query = f"SELECT timestamp, fast_fee, medium_fee, low_fee FROM {self.historical_table_name} ORDER BY timestamp"
            df = pd.read_sql(query, self.engine, parse_dates=['timestamp'])
            df['timestamp'] = df['timestamp'].dt.tz_localize(None).dt.tz_localize('UTC')  # Ensure UTC
            logging.info(f"Data fetched successfully from PostgreSQL table '{self.historical_table_name}'.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data from PostgreSQL database: {e}")
            return None

    @staticmethod
    def _create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates time-based features from a datetime index.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
            
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepares the data for model training. This function now returns the full dataset
        with features, so subsets can be taken based on lookback intervals later.
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True) # Ensure chronological order

        df = self._create_features(df)
        
        features = ['hour', 'dayofweek', 'month', 'year']
        target_fees = ['fast_fee', 'medium_fee', 'low_fee']
        
        X_all = df[features]
        y_all_dict = {fee: df[fee] for fee in target_fees if fee in df.columns}
        
        logging.info(f"Data preprocessed. Total observations: {len(X_all)}")
        return X_all, y_all_dict

    def _train_models(self, X_data: pd.DataFrame, y_data_dict: Dict[str, pd.Series], lookback_timedelta: Optional[pd.Timedelta] = None, model_key_name: Optional[str] = None) -> Dict[str, RandomForestRegressor]:
        """
        Trains a separate Random Forest Regressor model for each fee category,
        optionally filtering data by a lookback timedelta.
        """
        if lookback_timedelta:
            end_time = X_data.index.max()
            start_time = end_time - lookback_timedelta
            
            X_train_filtered = X_data[X_data.index >= start_time]
            y_train_dict_filtered = {fee: y_data_dict[fee][y_data_dict[fee].index >= start_time] 
                                     for fee in y_data_dict.keys()}
            
            logging.info(f"  Training with lookback: {str(lookback_timedelta)} (Data points: {len(X_train_filtered)})")
        else:
            X_train_filtered = X_data
            y_train_dict_filtered = y_data_dict
            logging.info(f"  Training with ALL available data (Data points: {len(X_train_filtered)})")

        if X_train_filtered.empty:
            logging.warning("  No data available for the specified lookback window. Skipping model training.")
            return {}

        trained_models = {}
        
        # Define default parameters in case a model_key_name or fee_type is not found in `parameters`
        default_params = {
            'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': None, 
            'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': True
        }

        for fee_type, y_train in y_train_dict_filtered.items():
            if not y_train.empty:
                logging.info(f"    Training model for {fee_type}...")
                
                current_model_params = default_params # Start with defaults

                # Use model_key_name to retrieve specific tuned parameters
                if model_key_name and model_key_name in parameters and fee_type in parameters[model_key_name]:
                    current_model_params = parameters[model_key_name][fee_type]
                    logging.info(f"    Using tuned parameters for {model_key_name} {fee_type}: {current_model_params}")
                else:
                    logging.warning(f"    No specific tuned parameters found for '{model_key_name}' and '{fee_type}'. Using default parameters.")
                
                model = RandomForestRegressor(**current_model_params, random_state=42, n_jobs=-1)
                model.fit(X_train_filtered, y_train)
                trained_models[fee_type] = model
                logging.info(f"    Model for {fee_type} trained.")
            else:
                logging.warning(f"    No target data for {fee_type} in this lookback. Skipping training.")
        return trained_models
    

    def _save_models(self, models: Dict[str, RandomForestRegressor], model_name_prefix: str) -> None:
        """Saves trained models to disk."""
        for fee_type, model in models.items():
            filename = os.path.join(self.model_dir, f"{model_name_prefix}_{fee_type}.pkl")
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                logging.info(f"  Saved model for {fee_type} as {filename}")
            except Exception as e:
                logging.error(f"  Failed to save model {filename}: {e}")

    def _load_models(self, model_name_prefix: str) -> Optional[Dict[str, RandomForestRegressor]]:
        """Loads trained models from disk."""
        loaded_models = {}
        for fee_type in ['fast_fee', 'medium_fee', 'low_fee']: # Assuming these fee types
            filename = os.path.join(self.model_dir, f"{model_name_prefix}_{fee_type}.pkl")
            if os.path.exists(filename):
                try:
                    with open(filename, 'rb') as f:
                        loaded_models[fee_type] = pickle.load(f)
                    logging.info(f"  Loaded model for {fee_type} from {filename}")
                except Exception as e:
                    logging.warning(f"  Could not load model {filename}: {e}. Will retrain.")
                    return None
            else:
                logging.info(f"  No saved model found for {fee_type} at {filename}. Will train new.")
                return None
        return loaded_models

    def _predict_future_fees(self, trained_models: Dict[str, RandomForestRegressor], current_time: datetime) -> pd.DataFrame:
        """
        Predicts future fees for a given time horizon and enforces logical ordering:
        low_fee <= medium_fee <= fast_fee.
        
        Args:
            trained_models (dict): A dictionary of trained models (one for each fee type).
            current_time (datetime): The timestamp from which to start the prediction horizon.
        
        Returns:
            pd.DataFrame: A DataFrame with the predicted fees, indexed by future timestamps,
                          with enforced logical ordering.
        """
        # Ensure we predict from the next full hour
        start_prediction_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        future_timestamps = pd.date_range(start=start_prediction_time, 
                                          periods=self.forecast_horizon_hours, 
                                          freq='h') 
        future_df = pd.DataFrame(index=future_timestamps)
        
        future_features_df = self._create_features(future_df)
        
        raw_predictions_df = pd.DataFrame(index=future_timestamps)
        
        for fee_type, model in trained_models.items():
            if model: # Ensure model was trained successfully for this fee type
                raw_predictions_df[fee_type] = model.predict(future_features_df)
            else:
                raw_predictions_df[fee_type] = np.nan # If no model was trained, fill with NaN
                
        # --- Enforce logical ordering: low_fee <= medium_fee <= fast_fee ---
        corrected_predictions_df = raw_predictions_df.copy()

        for index, row in raw_predictions_df.iterrows():
            if row.isnull().any(): # Skip if any value is NaN (e.g., model not trained)
                corrected_predictions_df.loc[index] = np.nan # Ensure corrected row is also NaN
                continue 
                
            low = row['low_fee']
            medium = row['medium_fee']
            fast = row['fast_fee']

            # Apply corrections in sequence
            low = min(low, medium, fast) # Low should be the minimum of the three
            fast = max(low, medium, fast) # Fast should be the maximum of the three
            # Medium should be between low and fast (if it's out of bounds, set to nearest boundary)
            medium = max(low, min(medium, fast))

            # Ensure non-negativity
            corrected_predictions_df.loc[index, 'low_fee'] = max(0, low)
            corrected_predictions_df.loc[index, 'medium_fee'] = max(0, medium)
            corrected_predictions_df.loc[index, 'fast_fee'] = max(0, fast)
            
        return corrected_predictions_df

    def _store_predictions_to_db(self, predictions_df: pd.DataFrame, model_name: str) -> None:
        """
        Stores the generated predictions into the database table 'fee_predictions_random_forest'.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame containing predictions with
                                           'fast_fee', 'medium_fee', 'low_fee' columns
                                           and a datetime index.
            model_name (str): The name of the model that generated these predictions (e.g., 'very_short').
        """
        records_to_insert = []

        for index, row in predictions_df.iterrows():
            if not row.isnull().any(): # Only store valid predictions
                records_to_insert.append({
                    'prediction_time': index,
                    'model_name': model_name,
                    'fast_fee': float(row['fast_fee']),    # Convert to standard Python float
                    'medium_fee': float(row['medium_fee']),# Convert to standard Python float
                    'low_fee': float(row['low_fee']),      # Convert to standard Python float
                    'generated_at': self.generated_at
                })
        
        if records_to_insert:
            conn = None
            try:
                conn = self.engine.connect()
                # Use SQLAlchemy core's insert() to insert multiple rows
                conn.execute(insert(self.fee_predictions_table), records_to_insert)
                conn.commit() # Commit the transaction
                logging.info(f"Successfully stored {len(records_to_insert)} predictions for model '{model_name}' to '{self.prediction_table_name}'.")
            except Exception as e:
                logging.error(f"Error storing predictions for model '{model_name}': {e}")
                if conn:
                    conn.rollback() # Rollback on error
            finally:
                if conn:
                    conn.close()
        else:
            logging.warning(f"No valid predictions to store for model '{model_name}'.")


    def tune_random_forest_hyperparameters(self, n_iter_search: int = 20, cv_folds: int = 5) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Performs hyperparameter tuning for RandomForestRegressor models for different fee types
        and lookback intervals using RandomizedSearchCV.

        Args:
            n_iter_search (int): Number of parameter settings that are sampled in RandomizedSearchCV.
                                Higher value means more exhaustive search but longer runtime.
            cv_folds (int): Number of cross-validation folds to use for robust evaluation.

        Returns:
            dict: A dictionary containing the best parameters found for each model and fee type.
                Example: {'very_short': {'fast_fee': {'n_estimators': 150, ...}, 'medium_fee': {...}}, ...}
        """
        logging.info(f"Starting hyperparameter tuning process at {datetime.now(timezone.utc)}...")

        # Fetch and preprocess all historical data
        df_all = self._fetch_fee_data()
        if df_all is None or df_all.empty:
            logging.error("No data fetched or data is empty. Cannot proceed with tuning.")
            return {}

        X_all, y_all_dict = self._preprocess_data(df_all)

        if X_all.empty:
            logging.error("Preprocessed data is empty. Cannot proceed with tuning.")
            return {}

        # Define the parameter distribution for RandomizedSearchCV
        param_dist = {
            'n_estimators': sp_randint(50, 300),
            'max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0],
            'max_depth': sp_randint(5, 50),
            'min_samples_split': sp_randint(2, 20),
            'min_samples_leaf': sp_randint(1, 10),
            'bootstrap': [True, False]
        }

        best_params_found = {}

        # Loop through each defined lookback interval
        for name, interval_str in self.lookback_intervals.items():
            logging.info(f"\n--- Tuning {name.replace('_', ' ').title()} Model ({interval_str} Lookback) ---")
            lookback_timedelta = pd.Timedelta(interval_str)

            # Filter data based on the current lookback interval
            end_time = X_all.index.max()
            start_time = end_time - lookback_timedelta
            
            X_train_filtered = X_all[X_all.index >= start_time]
            y_train_dict_filtered = {fee: y_all_dict[fee][y_all_dict[fee].index >= start_time] 
                                     for fee in y_all_dict.keys()}
            
            if X_train_filtered.empty:
                logging.warning(f"  No data available for lookback {interval_str}. Skipping tuning for {name}.")
                continue

            best_params_found[name] = {}

            # Tune for each fee type (fast_fee, medium_fee, low_fee)
            for fee_type, y_train in y_train_dict_filtered.items():
                if not y_train.empty:
                    logging.info(f"  Tuning for {fee_type}...")
                    
                    # Initialize a base model
                    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1) 

                    # Setup RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=rf_model, 
                        param_distributions=param_dist, 
                        n_iter=n_iter_search, 
                        cv=cv_folds, 
                        verbose=2, 
                        random_state=42, 
                        n_jobs=-1,
                        scoring='neg_mean_squared_error' 
                    )

                    # Fit the random search to the data
                    random_search.fit(X_train_filtered, y_train)

                    logging.info(f"  Best parameters for {fee_type}: {random_search.best_params_}")
                    logging.info(f"  Best (Negative) MSE for {fee_type}: {random_search.best_score_:.4f}")
                    logging.info(f"  Corresponding (Positive) MSE: {-random_search.best_score_:.4f}")

                    best_params_found[name][fee_type] = random_search.best_params_
                else:
                    logging.warning(f"  No target data for {fee_type} in this lookback. Skipping tuning.")

        logging.info("\nHyperparameter tuning completed.")
        return best_params_found

    def run(self, retrain_interval_hours: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Executes the full prediction pipeline: fetches data, trains/loads models for various
        lookback periods, predicts future fees, and stores the results.
        """
        logging.info(f"Starting FeePredictorRandomForest run at {datetime.now(timezone.utc)}...")
        
        current_time_for_prediction = datetime.now(timezone.utc)

        df_all = self._fetch_fee_data()
        if df_all is None or df_all.empty:
            logging.error("No data fetched or data is empty. Cannot proceed with prediction.")
            self.latest_predictions = {}
            return self.latest_predictions

        X_all, y_all_dict = self._preprocess_data(df_all)
        
        train_size = int(len(X_all) * 0.8)
        X_train_full = X_all[:train_size]
        y_train_full_dict = {key: val[:train_size] for key, val in y_all_dict.items()}
        
        X_test_full = X_all[train_size:]
        y_test_full_dict = {key: val[train_size:] for key, val in y_all_dict.items()}

        logging.info(f"Total historical data observations: {len(X_all)}")
        logging.info(f"Full training data observations (for lookback slicing): {len(X_train_full)}")
        logging.info(f"Test data observations (for evaluation): {len(X_test_full)}")

        self.latest_predictions = {} 
        self.trained_models_by_lookback = {} 

        for name, interval_str in self.lookback_intervals.items(): # 'name' is 'very_short', 'hourly', etc.
            logging.info(f"\n--- Processing {name.replace('_', ' ').title()} Model ({interval_str} Lookback) ---")
            lookback_timedelta = timedelta(hours=int(interval_str[:-1])) if interval_str[-1] == 'H' else timedelta(days=int(interval_str[:-1])) if interval_str[-1] == 'D' else timedelta(weeks=int(interval_str[:-1])) if interval_str[-1] == 'W' else timedelta(weeks=int(interval_str[:-1])*4) if interval_str[-1] == 'M' else None
            if lookback_timedelta is None:
                logging.error(f"Invalid interval_str: {interval_str}")
                continue
            
            model_name_prefix = f"model_{name}"
            trained_models_for_interval = None

            last_training_time_path = os.path.join(self.model_dir, f"{model_name_prefix}_last_trained.txt")
            retrain_needed = True
            if os.path.exists(last_training_time_path):
                try:
                    with open(last_training_time_path, 'r') as f:
                        last_trained_str = f.read().strip()
                        last_trained_time = datetime.fromisoformat(last_trained_str).astimezone(timezone.utc)
                        time_since_last_train = (current_time_for_prediction - last_trained_time).total_seconds() / 3600
                        if time_since_last_train < retrain_interval_hours:
                            logging.info(f"  Models for {name} trained {time_since_last_train:.1f} hours ago, less than {retrain_interval_hours} hours. Attempting to load existing models.")
                            loaded_models = self._load_models(model_name_prefix)
                            if loaded_models:
                                trained_models_for_interval = loaded_models
                                retrain_needed = False
                            else:
                                logging.warning(f"  Failed to load models for {name}. Retraining.")
                        else:
                            logging.info(f"  Models for {name} trained {time_since_last_train:.1f} hours ago, more than {retrain_interval_hours} hours. Retraining.")

                except Exception as e:
                    logging.warning(f"  Could not read last trained time for {name}: {e}. Retraining.")
            else:
                logging.info(f"  No previous training record found for {name}. Retraining.")

            if retrain_needed:
                # Step 3: Train models for the current lookback
                # PASSED 'name' as model_key_name HERE:
                trained_models_for_interval = self._train_models(X_train_full, y_train_full_dict, lookback_timedelta, model_key_name=name)
                if trained_models_for_interval:
                    self._save_models(trained_models_for_interval, model_name_prefix)
                    with open(last_training_time_path, 'w') as f:
                        f.write(current_time_for_prediction.isoformat())
                else:
                    logging.warning(f"  No models trained for {name} lookback after retraining attempt. Skipping prediction.")
                    continue

            if not trained_models_for_interval:
                logging.warning(f"  No trained models available for {name}. Skipping prediction and storage.")
                continue

            self.trained_models_by_lookback[name] = trained_models_for_interval 
            
            logging.info(f"Evaluating {name.replace('_', ' ').title()} model performance on the test set:")
            for fee_type, model in trained_models_for_interval.items():
                if model and not X_test_full.empty and not y_test_full_dict[fee_type].empty:
                    predictions_test = model.predict(X_test_full)
                    mse = mean_squared_error(y_test_full_dict[fee_type], predictions_test)
                    logging.info(f"  Mean Squared Error (MSE) for {fee_type}: {mse:.2f}")
                else:
                    logging.warning(f"  Skipping evaluation for {fee_type}: Test set is empty or model not trained.")
            
            current_predictions = self._predict_future_fees(trained_models_for_interval, current_time_for_prediction)
            self.latest_predictions[name] = current_predictions

            self._store_predictions_to_db(current_predictions, name)
        
        logging.info("FeePredictorRandomForest run completed. Predictions stored in DB and memory.")
        return self.latest_predictions