import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
import numpy as np
import logging
import os

# Scikit-learn imports for tuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV # Good for large search spaces
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import randint as sp_randint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper functions (copied from FeePredictor to make this script standalone) ---
def _fetch_fee_data(db_connection_string, historical_table_name):
    """
    Fetches fee data from the PostgreSQL database.
    Assumes the table has 'timestamp', 'fast_fee', 'medium_fee', and 'low_fee' columns.
    """
    try:
        engine = create_engine(db_connection_string)
        query = f"SELECT timestamp, fast_fee, medium_fee, low_fee FROM {historical_table_name} ORDER BY timestamp"
        df = pd.read_sql(query, engine)
        logging.info(f"Data fetched successfully from PostgreSQL table '{historical_table_name}'.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from PostgreSQL database: {e}")
        logging.error("Please ensure your DB_CONNECTION_STRING and historical_table_name are correct.")
        return None

def _create_features(df):
    """
    Creates time-based features from a datetime index.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index) # Ensure index is datetime
        
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df

def _preprocess_data(df):
    """
    Prepares the data for model training. This function now returns the full dataset
    with features, so subsets can be taken based on lookback intervals later.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True) # Ensure chronological order

    df = _create_features(df)
    
    features = ['hour', 'dayofweek', 'month', 'year']
    target_fees = ['fast_fee', 'medium_fee', 'low_fee']
    
    X_all = df[features]
    y_all_dict = {fee: df[fee] for fee in target_fees}
    
    logging.info(f"Data preprocessed. Total observations: {len(X_all)}")
    return X_all, y_all_dict


def tune_random_forest_hyperparameters(
    db_connection_string, 
    historical_table_name, 
    lookback_intervals=None,
    n_iter_search=20, # Number of parameter settings that are sampled (trade-off: time vs. comprehensiveness)
    cv_folds=5 # Number of cross-validation folds
):
    """
    Performs hyperparameter tuning for RandomForestRegressor models for different fee types
    and lookback intervals using RandomizedSearchCV.

    Args:
        db_connection_string (str): SQLAlchemy connection string for PostgreSQL.
        historical_table_name (str): Name of the historical fee data table.
        lookback_intervals (dict, optional): Dictionary of lookback intervals for tuning.
                                            Keys are model names (e.g., 'short_term'),
                                            values are pandas Timedelta strings (e.g., '3H').
                                            Defaults to {'short_term': '3H', 'medium_term': '3D', 'long_term': '3W'}.
        n_iter_search (int): Number of parameter settings that are sampled in RandomizedSearchCV.
                              Higher value means more exhaustive search but longer runtime.
        cv_folds (int): Number of cross-validation folds to use for robust evaluation.

    Returns:
        dict: A dictionary containing the best parameters found for each model and fee type.
              Example: {'short_term': {'fast_fee': {'n_estimators': 150, ...}, 'medium_fee': {...}}, ...}
    """
    logging.info(f"Starting hyperparameter tuning process at {dt.datetime.now()}...")

    if lookback_intervals is None:
        lookback_intervals = {
            "short_term": "3H",
            "medium_term": "3D",
            "long_term": "3W"
        }

    # Fetch and preprocess all historical data
    df_all = _fetch_fee_data(db_connection_string, historical_table_name)
    if df_all is None or df_all.empty:
        logging.error("No data fetched or data is empty. Cannot proceed with tuning.")
        return {}

    X_all, y_all_dict = _preprocess_data(df_all)

    if X_all.empty:
        logging.error("Preprocessed data is empty. Cannot proceed with tuning.")
        return {}

    # Define the parameter distribution for RandomizedSearchCV
    # These ranges are common starting points; adjust based on your data/needs.
    param_dist = {
        'n_estimators': sp_randint(50, 300), # Number of trees in the forest
        'max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0], # Number of features to consider when looking for the best split
        'max_depth': sp_randint(5, 50), # Maximum number of levels in tree
        'min_samples_split': sp_randint(2, 20), # Minimum number of samples required to split an internal node
        'min_samples_leaf': sp_randint(1, 10), # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False] # Whether bootstrap samples are used when building trees
    }

    best_params_found = {}

    # Loop through each defined lookback interval
    for name, interval_str in lookback_intervals.items():
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
                # Scoring: 'neg_mean_squared_error' is used because GridSearchCV/RandomizedSearchCV
                # tries to maximize the score, so we negate MSE to maximize performance.
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

# --- Example Usage for Tuning ---
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual PostgreSQL credentials
    DB_CONN_STR = 'postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain'
    HISTORICAL_FEE_TABLE = 'mempool_fee_histogram'

    # Define the lookback intervals for which you want to tune models
    # This should match (or be a subset of) what you use in your main FeePredictor
    tuning_lookbacks = {
        "very_short": "1H",
        "hourly": "12H",
        "daily": "7D",
        "weekly": "4W"
    }

    optimal_parameters = tune_random_forest_hyperparameters(
        db_connection_string=DB_CONN_STR,
        historical_table_name=HISTORICAL_FEE_TABLE,
        lookback_intervals=tuning_lookbacks,
        n_iter_search=30, # Increase for more thorough search, decrease for faster execution
        cv_folds=3 # Increase for more robust evaluation, decrease for faster execution
    )

    if optimal_parameters:
        logging.info("\n--- Summary of Optimal Hyperparameters Found ---")
        for model_name, fee_type_params in optimal_parameters.items():
            logging.info(f"\nModel: {model_name.replace('_', ' ').title()}")
            for fee_type, params in fee_type_params.items():
                logging.info(f"  {fee_type}: {params}")
    else:
        logging.info("No optimal parameters found due to data issues or tuning failures.")

    logging.info("\n-------------------------------------------------")
    logging.info("NEXT STEP: Take these 'best parameters' and update the RandomForestRegressor instantiation")
    logging.info("in your FeePredictor's _train_models method.")
    logging.info("For example, replace: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)")
    logging.info("With: RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, ..., random_state=42, n_jobs=-1)")