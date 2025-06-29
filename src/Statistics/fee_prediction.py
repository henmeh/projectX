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
import ast
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

warnings.filterwarnings("ignore", category=FutureWarning)

class HierarchicalGBR(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Regressor with fee hierarchy enforcement
    Ensures: fast_fee ≥ medium_fee ≥ low_fee
    """
    def __init__(self, **kwargs):
        self.base_estimator = GradientBoostingRegressor(**kwargs)
        self.estimator = None
        
    def fit(self, X, y):
        # Validate fee hierarchy in training data
        if not (y[:, 0] >= y[:, 1]).all() or not (y[:, 1] >= y[:, 2]).all():
            raise ValueError("Training data violates fee hierarchy")
            
        X, y = check_X_y(X, y, multi_output=True)
        self.estimator = self.base_estimator.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        y_pred = self.estimator.predict(X)
        
        # Enforce hierarchy
        for i in range(len(y_pred)):
            fast, medium, low = y_pred[i]
            
            # Correct ordering violations
            if fast < medium:
                adjustment = (medium - fast) / 2
                fast = medium + adjustment
                medium = medium - adjustment
                
            if medium < low:
                adjustment = (low - medium) / 2
                medium = low + adjustment
                low = low - adjustment
                
            if fast < medium or medium < low:
                # Fallback to proportional values
                if fast < low:
                    fast = max(fast, low * 1.5)
                medium = (fast + low) / 2
                low = min(medium, low)
                
            y_pred[i] = [fast, medium, low]
            
        return y_pred

class FeePredictor:
    def __init__(self, db_uri: str, model_path: str = 'fee_model.joblib', 
                 prediction_horizon: int = 1, retrain_interval: int = 10):
        self.db_uri = db_uri
        self.model_path = model_path
        self.prediction_horizon = prediction_horizon
        self.retrain_interval = retrain_interval
        self.engine = create_engine(db_uri, pool_size=10, max_overflow=20, pool_pre_ping=True)
        self.logger = self._setup_logger()
        self.model = None
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
            
            percentiles = sorted([pct_90, pct_75, pct_50], reverse=True)
            if percentiles != [pct_90, pct_75, pct_50]:
                self.logger.warning("Auto-corrected inverted percentiles: "
                                f"90p={pct_90}→{percentiles[0]}, "
                                f"75p={pct_75}→{percentiles[1]}, "
                                f"50p={pct_50}→{percentiles[2]}")
                pct_90, pct_75, pct_50 = percentiles
            
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
    
        
    def create_model(self) -> Pipeline:
        """Create constrained multi-output model"""
        numeric_features = self.feature_columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)])
        
        # Hierarchical model parameters
        params = {
            'loss': 'huber',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 7,
            'validation_fraction': 0.15,
            'n_iter_no_change': 15,
            'random_state': 42
        }
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', HierarchicalGBR(**params))
        ])
    
    
    def train_model(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[Pipeline, dict]:
        self.logger.info("Training hierarchical model with time-series validation")
        
        if features.empty or targets.empty:
            self.logger.error("Training aborted: Empty features/targets")
            return None, {}
        
        # Time-based cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        metrics = {}
        
        # Create the model
        model = self.create_model()
        
        # Prepare for cross-validation
        X = features.values
        y = targets.values
        
        # Initialize metrics storage
        for col in targets.columns:
            metrics[col] = {'mae': [], 'rmse': []}
        
        # Walk-forward validation
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics for each target
                for i, col in enumerate(targets.columns):
                    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                    metrics[col]['mae'].append(mae)
                    metrics[col]['rmse'].append(rmse)
                    
            except Exception as e:
                self.logger.error(f"Training fold failed: {str(e)}")
                continue
        
        # Final training on full dataset
        try:
            model.fit(X, y)
        except Exception as e:
            self.logger.error(f"Final training failed: {str(e)}")
            return None, {}
        
        # Calculate final metrics
        final_metrics = {}
        y_pred_full = model.predict(X)
        for i, col in enumerate(targets.columns):
            final_metrics[col] = {
                'mae': mean_absolute_error(y[:, i], y_pred_full[:, i]),
                'rmse': np.sqrt(mean_squared_error(y[:, i], y_pred_full[:, i])),
                'cv_mae': np.mean(metrics[col]['mae']),
                'cv_rmse': np.mean(metrics[col]['rmse'])
            }
        
        # Save model
        model_data = {
            'model': model,
            'feature_columns': self.feature_columns,
            'metrics': final_metrics,
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

        self.model = model
        self.last_trained = datetime.utcnow()
        self.model_version += 1
        
        self.logger.info(f"Training complete. Model version: {self.model_version - 1}")
        return model, final_metrics
    

    def load_model(self) -> bool:
        """
        Loads existing model. Returns True if successful.
        """
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
                
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.last_trained = model_data['trained_at']
            self.model_version = model_data['version']
            self.logger.info(f"Loaded model version {self.model_version}")
            return True
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return False


    def prepare_current_features(self, future_minutes: int = 0) -> Optional[pd.DataFrame]:
        """Prepare features for future prediction time"""
        self.logger.info("Preparing real-time features")
        
        # Calculate future timestamp
        prediction_time = datetime.utcnow() + timedelta(minutes=future_minutes)
        
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
        if len(df) == 0:
            return None
        latest = df.iloc[-1]
        
        # Prepare feature dictionary
        features = {
            'hour_sin': np.sin(2 * np.pi * prediction_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * prediction_time.hour / 24),
            'day_sin': np.sin(2 * np.pi * prediction_time.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * prediction_time.weekday() / 7),
            'block_speed': latest.get('block_speed', 1.0),
        }
        
        # Add all other features
        for col in self.feature_columns:
            if col in latest:
                features[col] = latest[col]
            else:
                features[col] = 0.0
                self.logger.warning(f"Feature {col} missing, using 0")
        
        feature_df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(feature_df.columns)
        for col in missing_cols:
            feature_df[col] = 0.0
            self.logger.warning(f"Added missing feature {col} with 0")
            
        return feature_df[self.feature_columns]    


    def predict_fees(self) -> Optional[Dict[str, float]]:
        """Predict fees for future horizon with hierarchy enforcement"""
        current_time = datetime.now()
        
        # Use cached prediction if available and recent
        if self.last_prediction:
            cache_age = (current_time - self.last_prediction['predicted_at']).total_seconds()
            if cache_age < self.prediction_horizon * 60 - 30:  # 30-second buffer
                return self.last_prediction
                
        # Load model if not available
        if not self.model:
            if not self.load_model():
                self.logger.error("Prediction aborted: No valid model")
                return None
                
        try:
            # Prepare features for exact prediction horizon
            features = self.prepare_current_features(future_minutes=self.prediction_horizon)
            if features is None or features.empty:
                self.logger.error("Feature preparation failed")
                return None
                
            # Make prediction
            predictions = self.model.predict(features.values)
            if predictions.size < 3:
                self.logger.error("Prediction output shape invalid")
                return None
                
            fast, medium, low = predictions[0]
            
            # Final validation and clamping
            if not (fast >= medium >= low):
                self.logger.warning(f"Hierarchy violation: {fast:.2f} ≥ {medium:.2f} ≥ {low:.2f}")
                # Enforce ordering
                medium = min(fast, max(medium, low))
                low = min(medium, low)
                fast = max(fast, medium)
                
                # Final check
                if not (fast >= medium >= low):
                    # Fallback: proportional values
                    fast = max(fast, 1.0)
                    low = min(low, fast * 0.8)
                    medium = (fast + low) / 2
            
            # Create prediction object
            prediction = {
                'predicted_at': current_time,
                'prediction_time': current_time + timedelta(minutes=self.prediction_horizon),
                'model_version': self.model_version,
                'fast_fee': max(0.1, fast),
                'medium_fee': max(0.05, medium),
                'low_fee': max(0.01, low),
            }
            
            # Update cache
            self.last_prediction = prediction
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}", exc_info=True)
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
                        """INSERT INTO fee_prediction 
                        (predicted_at, prediction_time, model_version, 
                         fast_fee_pred, medium_fee_pred, low_fee_pred)
                        VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            prediction["predicted_at"],
                            prediction["prediction_time"],
                            prediction["model_version"],
                            float(prediction["fast_fee"]),
                            float(prediction["medium_fee"]),
                            float(prediction["low_fee"])
                        )
                    )
                    conn.commit()
                    self.logger.info("Prediction saved successfully")
                    return True
                except Exception as e:
                    self.logger.error(f"Save failed: {e}")
                    conn.rollback()
                    return False
    
    def run_training(self, lookback_days: int = 1) -> bool:
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
            df = self.fetch_data(lookback_hours=lookback_days * 24)

            if len(df) < 500:
                self.logger.error("Insufficient training data")
                return False
                
            features, targets = self.preprocess_data(df)

            if len(features) < 300:
                self.logger.error("Insufficient features after preprocessing")
                return False
                
            model, metrics = self.train_model(features, targets)
            if not model:
                return False
                        
            # Log performance
            for target, metric in metrics.items():
                self.logger.info(
                    f"{target} - MAE: {metric['mae']:.4f}, "
                    f"RMSE: {metric['rmse']:.4f}, "
                    f"CV_MAE: {metric['cv_mae']:.4f}"
                )
                
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
                self.logger.error("Initial training failed. Creating fallback model")
                # Create fallback model
                self.model = self.create_model()
                self.last_trained = datetime.utcnow()
        
        # Service lifecycle
        self.logger.info("Starting prediction service")
        last_training = datetime.utcnow()
        next_prediction = datetime.utcnow()
        
        while True:
            try:
                now = datetime.utcnow()
                
                # Periodic retraining
                if (now - last_training).total_seconds() >= self.retrain_interval * 60:
                    self.logger.info("Starting periodic retraining")
                    if self.run_training():
                        last_training = datetime.utcnow()
                    else:
                        self.logger.warning("Retraining failed, using existing model")
                
                # Make prediction at scheduled time
                if now >= next_prediction:
                    prediction = self.predict_fees()
                    if prediction:
                        self.save_prediction(prediction)
                        self.logger.info(
                            f"Prediction for {prediction['prediction_time']}: "
                            f"Fast={prediction['fast_fee']:.2f} | "
                            f"Medium={prediction['medium_fee']:.2f} | "
                            f"Low={prediction['low_fee']:.2f}"
                        )
                    
                    # Schedule next prediction
                    next_prediction = now + timedelta(minutes=self.prediction_horizon)
                    
                    # Catch up if missed predictions
                    while next_prediction < now:
                        next_prediction += timedelta(minutes=self.prediction_horizon)
                
                # Calculate sleep time
                sleep_time = min(
                    max(1, (next_prediction - now).total_seconds()),
                    max(1, (last_training + timedelta(minutes=self.retrain_interval) - now).total_seconds())
                )
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("Service stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {str(e)}")
                time.sleep(min(30, self.prediction_horizon * 60))


if __name__ == "__main__":
    predictor = FeePredictor(
        db_uri="postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain",
        model_path="./models/fee_model_v1.joblib",
        prediction_horizon=10,  # Predict 10 minutes into future
        retrain_interval=1440   # Retrain every 24 hours
    )
    predictor.run()