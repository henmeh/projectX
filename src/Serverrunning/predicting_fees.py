"""
Based on random forest method (fee_predictor_random_forest.py) and facebook prohet (fee_predictor_fb_prophet.py)
every 25h fee predictions will be made for the next 48h into the future.
After 5 days model parameters for random forest will be recalculates for tuning. Stop this script update these parameters
to the class and restat this file by hand. For prophet tuning will be done automatically
"""

import time
import logging
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from FeePredictor.fee_predictor_random_forest import FeePredictorRandomForest
from FeePredictor.fee_predictor_fb_prophet import FeePredictorProphet

if __name__ == "__main__":
        # IMPORTANT: Replace with your actual PostgreSQL credentials
        DB_CONN_STR = 'postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain'
        HISTORICAL_FEE_TABLE = 'mempool_fee_histogram'
        PREDICTION_TABLE_RANDOM_FOREST = 'fee_predictions_random_forest' # Make sure this table exists and matches the schema below
        PREDICTION_TABLE_PROPHET = 'fee_predictions_prophet' # Make sure this table exists and matches the schema below


        custom_lookbacks_random_forest = {
        "very_short": "1H",
        "hourly": "12H",
        "daily": "7D",
        "weekly": "4W"
        }

        custom_lookbacks_prophet = {
        "very_short": "3d",  
        "hourly": "3w",      
        "daily": "180d",     
        "weekly": "365d"     
        }

        predictor_random_forest = FeePredictorRandomForest(
            db_connection_string=DB_CONN_STR,
            historical_table_name=HISTORICAL_FEE_TABLE,
            prediction_table_name=PREDICTION_TABLE_RANDOM_FOREST,
            lookback_intervals=custom_lookbacks_random_forest, 
            forecast_horizon_hours=48,
            model_dir='./trained_models_random_forest/'
        )

        predictor_prophet = FeePredictorProphet(
            db_connection_string=DB_CONN_STR,
            historical_table_name=HISTORICAL_FEE_TABLE,
            prediction_table_name=PREDICTION_TABLE_PROPHET,
            lookback_intervals=custom_lookbacks_prophet, 
            forecast_horizon_hours=48, # Predict 48 hours into the future
            model_dir='./trained_models_prophet/',
            tuning_enabled=True, 
            tuning_metrics=['mape'], # Mean Absolute Percentage Error
            tuning_cv_initial_days=60, # Initial training period for CV
            tuning_cv_period_days=15,  # Spacing between cutoff points
        )
        
        counter_for_model_tuning = 0

        while True:
            try:
                predictions_random_forest_result = predictor_random_forest.run(retrain_interval_hours=24) 
                predictions_prophet_result = predictor_prophet.run(retrain_interval_hours=24)

                if predictions_random_forest_result:
                    logging.info("\n--- Final Predictions Summary (from current run) ---")
                    for model_name, predictions_df in predictions_random_forest_result.items():
                        logging.info(f"\nPredictions from {model_name.replace('_', ' ').title()} Model:")
                        print(predictions_df) # Using print for DataFrame display
                        logging.info("-" * 50)
                else:
                    logging.info("No predictions generated.")

                if predictions_prophet_result:
                    logging.info("\n--- Final Predictions Summary (from current run) ---")
                    for model_name, predictions_df in predictions_prophet_result.items():
                        logging.info(f"\nPredictions from {model_name.replace('_', ' ').title()} Model (first 5 rows):")
                        print(predictions_df.head())
                        logging.info(f"... (total {len(predictions_df)} predictions for {model_name})")
                        logging.info("-" * 50)
                else:
                    logging.info("No predictions successfully generated and stored.")
                
                counter_for_model_tuning += 1

                if counter_for_model_tuning == 5:
                    optimal_parameters = predictor_random_forest.tune_random_forest_hyperparameters(n_iter_search=30, cv_folds=3)

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
             
                time.sleep(60 * 60 * 25)
            
            except RuntimeError as e:
                logging.critical(f"Application terminated due to critical setup error: {e}")
            except Exception as e:
                logging.critical(f"An unhandled error occurred during execution: {e}")
                logging.exception("Unhandled exception traceback:")