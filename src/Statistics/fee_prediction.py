import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine, text
import joblib
import logging
from datetime import datetime, timedelta
import time
import json
import psutil
import warnings
from typing import Tuple, Optional, Dict
import psycopg2
from psycopg2.extras import execute_values


warnings.filterwarnings("ignore", category=FutureWarning)

class FeePredictor:
    def __init__(self, db_uri: str, model_path: str = 'fee_model.joblib', 
                 prediction_horizon: int = 1, retrain_interval: int = 10):
        self.db_uri = db_uri
        self.model_path = model_path
        self.prediction_horizon = prediction_horizon
        self.retrain_interval = retrain_interval
        self.engine = create_engine(db_uri, pool_size=10, max_overflow=20, pool_pre_ping=True)
        self.logger = self._setup_logger()
        self.models = {}
        self.feature_columns = None
        self.last_trained = None
        self.model_version = 1
        self.histogram_cache = {}
        self.last_prediction = None
        self._ensure_model_dir()
        self.lags = [1, 5, 15, 30, 60]
        self.windows = [15, 30, 60]
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
        
    def connect_db(self):
        """Establish connection with optimized settings"""
        conn = psycopg2.connect(
            **self.db_params,
            application_name="BlockchainAnalytics",
            connect_timeout=10
        )
        
        # Set critical performance parameters
        with conn.cursor() as cur:
            try:
                # Stack depth solution for recursion errors
                cur.execute("SET max_stack_depth = '7680kB';")
                
                # Query optimization flags
                cur.execute("SET enable_partition_pruning = on;")
                cur.execute("SET constraint_exclusion = 'partition';")
                cur.execute("SET work_mem = '64MB';")
                
                # Transaction configuration
                cur.execute("SET idle_in_transaction_session_timeout = '5min';")
                conn.commit()
            except psycopg2.Error as e:
                print(f"Warning: Could not set session parameters: {e}")
                conn.rollback()
        
        return conn
    
    def _ensure_model_dir(self):
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                self.logger.info(f"Created model directory: {model_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create model directory: {str(e)}")
                self.model_path = os.path.basename(self.model_path)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('fee_predictor')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('fee_predictor.log')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
        
    def _db_safe_query(self, query: str) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn)
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            return pd.DataFrame()
        
    def fetch_data(self, lookback_hours: int = 168) -> pd.DataFrame:
        """
        Fetches data from the mempool_fee_histogram table for train the ml prediction model.
        """
        self.logger.info(f"Fetching fee data (last {lookback_hours} hours)")
        query = f"""
        SELECT timestamp, histogram, fast_fee, medium_fee, low_fee 
        FROM mempool_fee_histogram 
        WHERE timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
        ORDER BY timestamp
        """
        return self._db_safe_query(query)
    
    def safe_parse_histogram(self, hist_str: str) -> tuple[float, float, float, float, float]:
        if hist_str in self.histogram_cache:
            return self.histogram_cache[hist_str]
            
        default = (0.0, 0.0, 0.0, 0.0, 0.0)
        
        try:
            if isinstance(hist_str, str):
                try:
                    hist = json.loads(hist_str)
                except json.JSONDecodeError:
                    try:
                        hist = ast.literal_eval(hist_str)
                    except:
                        hist = []
            else:
                hist = hist_str
                
            if not isinstance(hist, list):
                return default
                
            fees, counts = [], []
            for item in hist:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        fee = float(item[0])
                        count = float(item[1])
                        if fee >= 0 and count >= 0:
                            fees.append(fee)
                            counts.append(count)
                    except (TypeError, ValueError):
                        continue
                elif isinstance(item, dict):
                    try:
                        fee = float(item.get('fee', 0))
                        count = float(item.get('count', 0))
                        if fee >= 0 and count >= 0:
                            fees.append(fee)
                            counts.append(count)
                    except (TypeError, ValueError):
                        continue
            
            if not fees:
                return default
                
            fees = np.array(fees)
            counts = np.array(counts)
            total = counts.sum()
            
            if total == 0:
                return default
                
            mean_fee = np.sum(fees * counts) / total
            fee_distribution = np.repeat(fees, counts.astype(int))
            
            if len(fee_distribution) < 10:
                pct_90 = np.percentile(fees, 90)
                pct_75 = np.percentile(fees, 75)
                pct_50 = np.percentile(fees, 50)
            else:
                pct_90 = np.percentile(fee_distribution, 90)
                pct_75 = np.percentile(fee_distribution, 75)
                pct_50 = np.percentile(fee_distribution, 50)
            
            result = (total, mean_fee, pct_90, pct_75, pct_50)
            self.histogram_cache[hist_str] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Histogram parse error: {str(e)}")
            return default
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Centralized feature engineering"""
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Block time features
        df['block_interval'] = df.index.to_series().diff().dt.total_seconds().fillna(600)
        df['block_speed'] = 600 / df['block_interval']
        
        # Lag features
        for lag in self.lags:
            df[f'fast_fee_lag_{lag}'] = df['fast_fee'].shift(lag)
            df[f'medium_fee_lag_{lag}'] = df['medium_fee'].shift(lag)
            df[f'low_fee_lag_{lag}'] = df['low_fee'].shift(lag)
            df[f'total_tx_lag_{lag}'] = df['total_tx'].shift(lag)
        
        # Rolling features
        for window in self.windows:
            df[f'fast_fee_rolling_{window}'] = df['fast_fee'].rolling(f'{window}min').mean()
            df[f'tx_rolling_{window}'] = df['total_tx'].rolling(f'{window}min').mean()
        
        # EMA features
        df['fast_fee_ema_30'] = df['fast_fee'].ewm(span=30, adjust=False).mean()
        df['tx_ema_30'] = df['total_tx'].ewm(span=30, adjust=False).mean()
        
        return df.fillna(0)
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Preprocessing data")
        
        if df.empty or len(df) < 100:
            self.logger.warning("Insufficient data for preprocessing")
            return pd.DataFrame(), pd.DataFrame()

        df = df.set_index('timestamp').sort_index()
        # Parse histogram
        df['histogram_features'] = df['histogram'].apply(self.safe_parse_histogram)
        df[['total_tx', 'mean_fee', 'p90_fee', 'p75_fee', 'p50_fee']] = pd.DataFrame(
            df['histogram_features'].tolist(), index=df.index
        )
        
        # Add all features
        df = self._add_features(df)
        
        # Future targets
        df['fast_fee_future'] = df['fast_fee'].shift(-self.prediction_horizon)
        df['medium_fee_future'] = df['medium_fee'].shift(-self.prediction_horizon)
        df['low_fee_future'] = df['low_fee'].shift(-self.prediction_horizon)
        
        # Drop initial NaNs
        df = df.dropna()
        
        # Identify feature columns
        self.feature_columns = df.columns.difference([
            'histogram', 'histogram_features', 'fast_fee', 'medium_fee', 'low_fee',
            'fast_fee_future', 'medium_fee_future', 'low_fee_future'
        ]).tolist()
        
        # Prepare features and targets
        features = df[self.feature_columns]
        targets = df[['fast_fee_future', 'medium_fee_future', 'low_fee_future']]

        return features, targets
    
        
    def create_model(self, target_name: str = 'fast_fee') -> Pipeline:
        # Use all available features
        numeric_features = self.feature_columns
        
        # Feature-specific hyperparameters
        hyperparams = {
            'fast_fee': {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 7},
            'medium_fee': {'n_estimators': 250, 'learning_rate': 0.07, 'max_depth': 6},
            'low_fee': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5}
        }
        
        params = hyperparams.get(target_name, hyperparams['fast_fee'])
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                loss='huber',
                random_state=42,
                validation_fraction=0.15,
                n_iter_no_change=15,
                **params
            ))
        ])
    
    def train_model(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[dict, dict]:
        self.logger.info("Training models with time-series validation")
        
        if features.empty or targets.empty:
            self.logger.error("Training aborted: Empty features/targets")
            return {}, {}
        
        # Time-based cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        metrics = {target: [] for target in targets.columns}
        models = {}
        
        # Train separate models for each target
        for target_col in targets.columns:
            target_name = target_col.replace('_future', '')
            self.logger.info(f"Training model for {target_name}")
            
            fold_metrics = {'mae': [], 'rmse': []}
            model = self.create_model(target_name)
            
            # Walk-forward validation
            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train = targets[target_col].iloc[train_idx]
                y_test = targets[target_col].iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                fold_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
                fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            
            # Store model and metrics
            models[target_name] = model
            metrics[target_col] = {
                'mae': np.mean(fold_metrics['mae']),
                'rmse': np.mean(fold_metrics['rmse']),
                'fold_mae': fold_metrics['mae'],
                'fold_rmse': fold_metrics['rmse']
            }
        
        # Save model ensemble
        model_data = {
            'models': models,
            'feature_columns': self.feature_columns,
            'metrics': metrics,
            'version': self.model_version,
            'trained_at': datetime.utcnow()
        }
        
        try:
            joblib.dump(model_data, self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            fallback_path = f"fee_model_v{self.model_version}.joblib"
            joblib.dump(model_data, fallback_path)
            self.model_path = fallback_path
            self.logger.warning(f"Saved model to fallback location: {fallback_path}")

        self.models = models
        self.last_trained = datetime.utcnow()
        self.model_version += 1
        
        self.logger.info(f"Training complete. Model version: {self.model_version - 1}")
        return models, metrics
    
    def load_model(self) -> bool:
        """
        Loads existing model. Returns True if there is a model, returns False if there is none or one that could not be loaded.
        """
        try:
            model_data = joblib.load(self.model_path)
            self.models = model_data['models']
            self.feature_columns = model_data['feature_columns']
            self.last_trained = model_data['trained_at']
            self.model_version = model_data['version']
            self.logger.info(f"Loaded model version {self.model_version}")
            return True
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {self.model_path}")
            return False
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return False
    
    def prepare_current_features(self) -> Optional[pd.DataFrame]:
        self.logger.info("Preparing real-time features")
        
        # Fetch recent data (last 2 hours)
        query = """
        SELECT timestamp, histogram, fast_fee, medium_fee, low_fee 
        FROM mempool_fee_histogram 
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
        ORDER BY timestamp DESC
        """
        try:
            df = pd.read_sql(text(query), self.engine)
        except Exception as e:
            self.logger.error(f"Data fetch failed: {str(e)}")
            return None
        
        if df.empty:
            self.logger.warning("No recent data available")
            return None
        
        # Preprocessing pipeline
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Parse histogram
        df['histogram_features'] = df['histogram'].apply(self.safe_parse_histogram)
        df[['total_tx', 'mean_fee', 'p90_fee', 'p75_fee', 'p50_fee']] = pd.DataFrame(
            df['histogram_features'].tolist(), index=df.index
        )
        
        # Add all features consistently with training
        df = self._add_features(df)
        
        # Use the latest data point
        latest = df.iloc[-1]
        
        # Prepare feature dictionary
        now = datetime.utcnow()
        features = {
            'hour_sin': np.sin(2 * np.pi * now.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24),
            'day_sin': np.sin(2 * np.pi * now.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * now.weekday() / 7),
            'block_speed': latest['block_speed'],
        }
        
        # Add all other features
        for col in self.feature_columns:
            if col in latest.index:
                features[col] = latest[col]
            else:
                features[col] = 0.0
                self.logger.warning(f"Feature {col} missing in real-time data, using 0")
        
        feature_df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        return feature_df[self.feature_columns]
    
    def predict_fees(self) -> Optional[dict[str, float]]:
        if self.last_prediction and (datetime.now() - self.last_prediction['timestamp']).seconds < 60:
            return self.last_prediction
        
        if not self.models:
            self.logger.warning("Models not loaded. Attempting load...")
            if not self.load_model():
                self.logger.error("Prediction aborted: No valid models")
                return None
                
        features = self.prepare_current_features()
        if features is None or features.empty:
            return None
            
        try:
            prediction = {
                'timestamp': datetime.now(),
                'model_version': self.model_version
            }
            
            for fee_type, model in self.models.items():
                prediction[f'{fee_type}'] = max(0, model.predict(features)[0])
            
            self.last_prediction = prediction
            return prediction
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return None
    
    def save_prediction(self, prediction: dict) -> bool:
        """
        Saves fee prediction in database
        """
        if not prediction:
            return False
        
        with self.connect_db() as conn:
            with conn.cursor() as cursor:                
                try:
                    cursor.execute(
                                   """INSERT INTO fee_prediction (prediction_timestamp, model_version, fast_fee_pred, medium_fee_pred, low_fee_pred)
                                    VALUES (%s, %s, %s, %s, %s)""",
                                    (prediction["timestamp"], prediction["model_version"], float(prediction["fast_fee"]), float(prediction["medium_fee"]), float(prediction["low_fee"])))
                    conn.commit()
                    self.logger.info("Prediction saved successfully")
                    return True
                except Exception as e:
                    self.logger.error(f"Save failed: {e}")
                    return False
                
    
    def run_training(self, lookback_days:int = 1) -> bool:
        """
        Trains the ML prediction model with data from the DB.
        """
        try:
            # Check system resources
            mem = psutil.virtual_memory()
            if mem.available < 1 * 1024**3:  # 1GB threshold
                self.logger.warning("Insufficient memory for training")
                return False
                
            start_time = time.time()
            df = self.fetch_data(lookback_hours=lookback_days*24)

            if len(df) < 500:
                self.logger.error("Insufficient training data")
                return False
                
            features, targets = self.preprocess_data(df)

            if len(features) < 300:
                self.logger.error("Insufficient features after preprocessing")
                return False
                
            _, metrics = self.train_model(features, targets)
                        
            # Log performance
            for target, metric in metrics.items():
                self.logger.info(f"{target} - MAE: {metric['mae']:.2f}, RMSE: {metric['rmse']:.2f}")
                
            self.logger.info(f"Training completed in {(time.time()-start_time)/60:.1f} minutes")
            
            return True
        except Exception as e:
            self.logger.exception(f"Training failed: {str(e)}")
            return False
    
    def run(self) -> None:
        # Initial model loading
        if not self.load_model():
            self.logger.info("No model found. Starting initial training...")
            if not self.run_training():
                self.logger.error("Initial training failed. Using fallback models")
                
                self.models = {
                    'fast_fee': self.create_model('fast_fee'),
                    'medium_fee': self.create_model('medium_fee'),
                    'low_fee': self.create_model('low_fee')
                }
        
        # Service lifecycle
        self.logger.info("Starting prediction service")
        last_training = datetime.now()
        last_prediction_time = datetime.now()    
        
        while True:
            try:
        
                cycle_start = time.time()
                current_time = datetime.now()

                # Periodic retraining
                if (current_time - last_training).total_seconds() >= self.retrain_interval * 60:
                    self.logger.info("Starting periodic retraining")
                    if self.run_training():
                        last_training = datetime.now()
                    else:
                        self.logger.warning("Retraining failed, using existing model")
                
                # Make prediction at regular intervals
                if (current_time - last_prediction_time).total_seconds() >= self.prediction_horizon * 60:
                    prediction = self.predict_fees()
                    if prediction:
                        self.logger.info("Successful fee prediction")
                        self.save_prediction(prediction)
                        self.logger.info(
                            f"Prediction: Fast={prediction.get('fast_fee', 0):.1f} | "
                            f"Medium={prediction.get('medium_fee', 0):.1f} | "
                            f"Low={prediction.get('low_fee', 0):.1f}"
                        )
                        last_prediction_time = current_time
                    
                # Dynamic sleep adjustment
                cycle_time = time.time() - cycle_start
                sleep_time = max(1, self.prediction_horizon * 60 - cycle_time)
                time.sleep(sleep_time)
                #time.sleep(60)
                
            except KeyboardInterrupt:
                self.logger.info("Service stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {str(e)}")
                time.sleep(30)  # Cool-down period


if __name__ == "__main__":
    predictor = FeePredictor(
        db_uri="postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain",
        model_path="./models/fee_model_v1.joblib",
        prediction_horizon=10,#10,
        retrain_interval=100#1440  # Daily retraining
    )
    predictor.run()