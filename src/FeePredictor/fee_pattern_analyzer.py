import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime
import psycopg2
from psycopg2 import Error as Psycopg2Error
import joblib # For saving/loading models

class FeePatternAnalyzer:
    """
    A class to analyze historical Bitcoin transaction fee data, find patterns using K-Means clustering,
    and provide predictions and recommendations for optimal transaction times.
    """
    def __init__(self, db_config, data_interval='6 months', n_clusters=3, model_path='fee_model.pkl', scaler_path='fee_scaler.pkl', category_map_path='fee_category_map.pkl'):
        """
        Initializes the FeePatternAnalyzer.

        Args:
            db_config (dict): Dictionary containing PostgreSQL database connection parameters.
            data_interval (str): The time interval for fetching historical data (e.g., '3 months', '6 months').
            n_clusters (int): The number of clusters for K-Means (e.g., 3 for Low, Medium, High fees).
            model_path (str): File path to save/load the KMeans model.
            scaler_path (str): File path to save/load the StandardScaler.
            category_map_path (str): File path to save/load the cluster ID to category mapping.
        """
        self.db_config = db_config
        self.data_interval = data_interval
        self.n_clusters = n_clusters
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.category_map_path = category_map_path

        self.kmeans_model = None
        self.scaler = None
        self.feature_cols = None
        self.df_with_categories = None # Stores the historical DataFrame with cluster categories
        self.cluster_id_to_category = None # Maps cluster IDs to human-readable categories
        self.cluster_summary = None # Summary of cluster characteristics

        # Day names for display, matching the 0=Sunday, 6=Saturday convention from DB
        self.day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    def _load_and_prepare_data_from_db(self):
        """
        Connects to PostgreSQL using `self.db_config`, executes the specified query
        for `self.data_interval`, and prepares the data for analysis.
        Ensures all 7 days * 24 hours slots are present, filling missing avg_fee with 0.

        Returns:
            pandas.DataFrame: Prepared DataFrame with historical fee data.
        Raises:
            Exception: If there's an error connecting to the database or executing the query.
        """
        conn = None
        df = pd.DataFrame()
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            query = """
                SELECT
                    EXTRACT(DOW FROM timestamp) AS day_of_week_num, -- 0=Sunday, 1=Monday, ..., 6=Saturday
                    EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC') AS hour_of_day,
                    CAST(AVG(fast_fee) AS NUMERIC(10,1)) AS avg_fee
                FROM
                    mempool_fee_histogram
                WHERE
                    timestamp >= NOW() - INTERVAL %s
                GROUP BY
                    EXTRACT(DOW FROM timestamp),
                    EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC')
                ORDER BY
                    day_of_week_num,
                    hour_of_day;
            """
            cursor.execute(query, (self.data_interval,))
            records = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(records, columns=col_names)

            if not all(col in df.columns for col in ['day_of_week_num', 'hour_of_day', 'avg_fee']):
                raise ValueError("Query result must contain 'day_of_week_num', 'hour_of_day', 'avg_fee' columns.")

            df['day_of_week_num'] = df['day_of_week_num'].astype(int)
            df['hour_of_day'] = df['hour_of_day'].astype(int)
            df['avg_fee'] = df['avg_fee'].astype(float)

            # Ensure all 7 days * 24 hours are present, filling gaps with 0
            all_time_slots = pd.MultiIndex.from_product(
                [range(7), range(24)], names=['day_of_week_num', 'hour_of_day']
            ).to_frame(index=False)

            df = pd.merge(all_time_slots, df, on=['day_of_week_num', 'hour_of_day'], how='left')
            df['avg_fee'] = df['avg_fee'].fillna(0) # Fill NaN from left join with 0

            df['time_slot_id'] = df.apply(lambda row: f"{row['day_of_week_num']}-{row['hour_of_day']}", axis=1)

        except (Exception, Psycopg2Error) as error:
            print(f"Error while connecting to PostgreSQL or executing query: {error}")
            raise # Re-raise the exception to be caught by the caller
        finally:
            if conn:
                cursor.close()
                conn.close()
        return df

    def _create_features(self, df):
        """
        Creates features suitable for clustering, including cyclical features for hour and day.

        Args:
            df (pandas.DataFrame): The input DataFrame with 'day_of_week_num', 'hour_of_day', 'avg_fee'.

        Returns:
            tuple: (scaled_features, scaler, feature_column_names)
        """
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)

        features_df = df[['avg_fee', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        return scaled_features, scaler, features_df.columns

    def _perform_kmeans_clustering(self, scaled_data):
        """
        Performs K-Means clustering on the scaled data.

        Args:
            scaled_data (np.array): The feature data after scaling.

        Returns:
            sklearn.cluster.KMeans: The fitted KMeans model.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10) # n_init for robust initialization
        kmeans.fit(scaled_data)
        return kmeans

    def _interpret_clusters(self, df, kmeans_model, feature_names):
        """
        Interprets the clusters by calculating the average fee for each cluster and
        assigning meaningful categories ('Low Fee', 'Medium Fee', 'High Fee').

        Args:
            df (pandas.DataFrame): The original DataFrame.
            kmeans_model (sklearn.cluster.KMeans): The fitted KMeans model.
            feature_names (list): List of feature column names used for clustering.

        Returns:
            tuple: (df_with_categories, cluster_summary, cluster_id_to_category_map)
        """
        df['cluster_label'] = kmeans_model.labels_

        cluster_fee_means = df.groupby('cluster_label')['avg_fee'].mean().sort_values()

        # Dynamic categorization based on number of clusters
        fee_categories = []
        if self.n_clusters == 2:
            fee_categories = ['Low Fee', 'High Fee']
        elif self.n_clusters == 3:
            fee_categories = ['Low Fee', 'Medium Fee', 'High Fee']
        elif self.n_clusters > 3:
            fee_categories = [f'Fee Category {i+1}' for i in range(self.n_clusters)]
        else: # Should not happen with n_clusters >= 2
             fee_categories = [f'Category {i+1}' for i in range(self.n_clusters)]


        cluster_to_category = {cluster_id: category for cluster_id, category in zip(cluster_fee_means.index, fee_categories)}
        df['fee_category'] = df['cluster_label'].map(cluster_to_category)

        cluster_avg_fees = df.groupby('cluster_label')['avg_fee'].mean()
        cluster_info = {
            cluster_to_category[idx]: {
                'avg_fee': cluster_avg_fees[idx],
                'times_in_category': df[df['cluster_label'] == idx][['day_of_week_num', 'hour_of_day']].drop_duplicates().to_dict(orient='records')
            }
            for idx in cluster_fee_means.index
        }
        return df, cluster_info, cluster_to_category

    def _get_fee_prediction_raw(self, day_of_week_num, hour_of_day):
        """
        Internal helper to get the raw cluster ID prediction for a given time.

        Args:
            day_of_week_num (int): Day of the week (0=Sunday, 6=Saturday).
            hour_of_day (int): Hour of the day (0-23 UTC).

        Returns:
            int: The predicted cluster ID.
        Raises:
            RuntimeError: If the model components are not loaded or trained.
        """
        if self.kmeans_model is None or self.scaler is None or self.feature_cols is None or self.df_with_categories is None:
            raise RuntimeError("Model is not trained or loaded. Call .run() first (with train_model=True or False).")

        # Use the historical average fee for that specific time slot as a placeholder for prediction
        # This helps the prediction be consistent with the historical patterns the model learned.
        historical_avg_fee_for_slot = self.df_with_categories[
            (self.df_with_categories['day_of_week_num'] == day_of_week_num) &
            (self.df_with_categories['hour_of_day'] == hour_of_day)
        ]['avg_fee'].mean()

        # Fallback to overall historical average fee if specific slot data is missing
        if pd.isna(historical_avg_fee_for_slot):
            placeholder_avg_fee = self.df_with_categories['avg_fee'].mean()
        else:
            placeholder_avg_fee = historical_avg_fee_for_slot

        # Create the input features, including cyclical transformations
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        day_sin = np.sin(2 * np.pi * day_of_week_num / 7)
        day_cos = np.cos(2 * np.pi * day_of_week_num / 7)

        input_data = pd.DataFrame([[placeholder_avg_fee, hour_sin, hour_cos, day_sin, day_cos]],
                                  columns=self.feature_cols)

        # Scale the input data using the same scaler fitted on training data
        scaled_input = self.scaler.transform(input_data)

        # Predict the cluster
        predicted_cluster = self.kmeans_model.predict(scaled_input)[0]
        return predicted_cluster

    def save_model(self):
        """Saves the trained model components (KMeans model, StandardScaler, category map) to disk."""
        try:
            joblib.dump(self.kmeans_model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.cluster_id_to_category, self.category_map_path)
            print(f"Model saved to {self.model_path}, {self.scaler_path}, {self.category_map_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """
        Loads the trained model components from disk.
        Returns True if successful, False otherwise.
        """
        try:
            self.kmeans_model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.cluster_id_to_category = joblib.load(self.category_map_path)
            print("Model loaded successfully.")
            return True
        except FileNotFoundError:
            print("Model files not found. Please train the model first or check paths.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def run(self, train_model=True):
        """
        Starts the complete fee pattern analysis process. This is the main entry point.

        Args:
            train_model (bool): If True, trains a new model from database data and saves it.
                                If False, attempts to load an existing model.
                                In either case, it fetches historical data to ensure
                                `self.df_with_categories` is populated for predictions.

        Returns:
            bool: True if the process was successful (model trained/loaded), False otherwise.
        """
        if train_model:
            print("--- Starting Fee Pattern Analysis (Training New Model) ---")
            try:
                print(f"Loading data for the last {self.data_interval} from database...")
                df_historical = self._load_and_prepare_data_from_db()
                if df_historical.empty:
                    print("No data retrieved from the database. Cannot train model.")
                    return False
                # df_with_categories will be populated by _interpret_clusters
                # We need a copy of df_historical to ensure _interpret_clusters doesn't modify
                # the one used for feature creation directly if it's reused.
                temp_df_for_interpretation = df_historical.copy()

                print("Creating features...")
                scaled_features, self.scaler, self.feature_cols = self._create_features(df_historical)

                print(f"Performing K-Means Clustering with {self.n_clusters} clusters...")
                self.kmeans_model = self._perform_kmeans_clustering(scaled_features)

                print("Interpreting clusters and categorizing time slots...")
                self.df_with_categories, self.cluster_summary, self.cluster_id_to_category = self._interpret_clusters(
                    temp_df_for_interpretation, self.kmeans_model, self.feature_cols
                )

                print("\n--- Model Training Complete ---")
                for category, info in self.cluster_summary.items():
                    print(f"- {category}: Average Fee = {info['avg_fee']:.2f} sat/vB")
                print("Model trained and ready for predictions.")
                self.save_model() # Save the newly trained model
                return True

            except Exception as e:
                print(f"An error occurred during model training: {e}")
                return False
        else:
            print("--- Attempting to Load Existing Fee Pattern Model ---")
            success = self.load_model()
            if success:
                # Even when loading, we need the df_with_categories for _get_fee_prediction_raw's
                # historical_avg_fee_for_slot lookup. So, we re-fetch it.
                print("Re-fetching historical data for prediction context (needed for placeholder fee logic)...")
                try:
                    df_historical_for_load_context = self._load_and_prepare_data_from_db()
                    # After loading, interpret clusters again to populate df_with_categories
                    # This relies on the loaded kmeans_model and features
                    # This assumes the features (avg_fee, cyclical) from df_historical_for_load_context
                    # will yield the same cluster labels as when trained.
                    # This might be slightly off if new data significantly changes underlying patterns
                    # but model isn't retrained. For strict consistency, df_with_categories should also be saved.
                    # For now, let's derive it:
                    scaled_features_for_load_context, _, _ = self._create_features(df_historical_for_load_context.copy())
                    self.df_with_categories, self.cluster_summary, self.cluster_id_to_category = self._interpret_clusters(
                        df_historical_for_load_context.copy(), self.kmeans_model, self.feature_cols
                    )

                except Exception as e:
                    print(f"Warning: Could not re-fetch/derive historical data for prediction context: {e}")
                    print("Predictions might use a less accurate global average fee if specific historical slot data is missing.")
            return success

    def predict_fee_category(self, day_of_week_num, hour_of_day):
        """
        Predicts the fee category ('Low Fee', 'Medium Fee', 'High Fee') for a specific time.
        Requires the model to be trained or loaded first (by calling `run()`).

        Args:
            day_of_week_num (int): Day of the week (0=Sunday, 6=Saturday).
            hour_of_day (int): Hour of the day (0-23 UTC).

        Returns:
            str: The predicted fee category, or "Unknown" if mapping fails.
        """
        try:
            predicted_cluster_id = self._get_fee_prediction_raw(day_of_week_num, hour_of_day)
            return self.cluster_id_to_category.get(predicted_cluster_id, "Unknown")
        except RuntimeError as e:
            print(f"Prediction error: {e}")
            return "Error: Model not ready"
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            return "Error: Prediction failed"

    def get_low_fee_recommendations(self, num_hours_ahead=24):
        """
        Provides recommendations for 'Low Fee' transaction times in the upcoming hours,
        based on historical patterns identified by the model.
        Requires the model to be trained or loaded first (by calling `run()`).

        Args:
            num_hours_ahead (int): How many hours into the future to check for recommendations.

        Returns:
            list: A list of strings, each describing a recommended low-fee time slot.
                  Returns an empty list if no low-fee category is found or an error occurs.
        """
        if self.cluster_id_to_category is None or self.cluster_summary is None or self.df_with_categories is None:
            print("Model is not trained or loaded. Cannot provide recommendations. Call .run() first.")
            return []

        low_fee_cluster_id = None
        for cluster_id, category in self.cluster_id_to_category.items():
            if category == 'Low Fee':
                low_fee_cluster_id = cluster_id
                break

        if low_fee_cluster_id is None:
            return ["No 'Low Fee' category identified by the model."]

        # Get historical average fees for low fee slots for display
        low_fee_slots_data = self.df_with_categories[self.df_with_categories['cluster_label'] == low_fee_cluster_id].copy()

        recommended_future_times = []
        now_utc = datetime.datetime.utcnow()
        current_day_num_db_format = (now_utc.weekday() + 1) % 7 # Convert Python's Monday=0 to DB's Sunday=0

        for i in range(num_hours_ahead):
            check_time_utc = now_utc + datetime.timedelta(hours=i)
            check_day_num_db_format = (check_time_utc.weekday() + 1) % 7
            check_hour = check_time_utc.hour

            try:
                predicted_cluster_id_for_future = self._get_fee_prediction_raw(
                    check_day_num_db_format, check_hour
                )
            except RuntimeError as e:
                print(f"Warning: Prediction error for {self.day_names[check_day_num_db_format]} {str(check_hour).zfill(2)}:00 UTC - {e}")
                continue # Skip this time slot if model not ready

            if predicted_cluster_id_for_future == low_fee_cluster_id:
                # Find the average fee for this specific slot from the historical data for display
                avg_fee_for_display = low_fee_slots_data[
                    (low_fee_slots_data['day_of_week_num'] == check_day_num_db_format) &
                    (low_fee_slots_data['hour_of_day'] == check_hour)
                ]['avg_fee'].mean()

                fee_str = f"Avg Fee: {avg_fee_for_display:.1f} sat/vB" if not pd.isna(avg_fee_for_display) else "Fee: N/A"

                recommended_future_times.append(
                    f"{self.day_names[check_day_num_db_format]} {str(check_hour).zfill(2)}:00 UTC ({fee_str})"
                )
        return recommended_future_times

    def get_overall_fee_patterns(self):
        """
        Summarizes the identified fee patterns (Low, Medium, High) into human-readable descriptions.
        This provides the "Wednesday afternoon is cheap" type of information.

        Returns:
            dict: A dictionary where keys are fee categories ('Low Fee', 'Medium Fee', 'High Fee')
                  and values are lists of descriptive strings for when those patterns typically occur.
        """
        if self.cluster_summary is None or self.df_with_categories is None:
            print("Model is not trained or loaded. Cannot provide overall fee patterns. Call .run() first.")
            return {}

        patterns_summary = {}

        for category, info in self.cluster_summary.items():
            avg_fee_for_category = info['avg_fee']
            times = pd.DataFrame(info['times_in_category'])
            if times.empty:
                patterns_summary[category] = [f"No specific times identified for this category (Avg Fee: {avg_fee_for_category:.2f} sat/vB)."]
                continue

            # Sort by day and hour for continuous blocks
            times = times.sort_values(by=['day_of_week_num', 'hour_of_day']).reset_index(drop=True)

            day_patterns = {}
            for day_num in range(7):
                day_data = times[times['day_of_week_num'] == day_num]['hour_of_day'].tolist()
                if not day_data:
                    continue

                # Find continuous blocks of hours
                blocks = []
                if day_data:
                    current_block_start = day_data[0]
                    current_block_end = day_data[0]
                    for i in range(1, len(day_data)):
                        if day_data[i] == current_block_end + 1:
                            current_block_end = day_data[i]
                        else:
                            blocks.append((current_block_start, current_block_end))
                            current_block_start = day_data[i]
                            current_block_end = day_data[i]
                    blocks.append((current_block_start, current_block_end))

                block_strings = []
                for start, end in blocks:
                    if start == end:
                        block_strings.append(f"{str(start).zfill(2)}:00 UTC")
                    else:
                        block_strings.append(f"{str(start).zfill(2)}:00-{str(end+1).zfill(2)}:00 UTC") # +1 for end hour (e.g. 17:00-18:00 means hour 17)
                day_patterns[self.day_names[day_num]] = ", ".join(block_strings)

            category_descriptions = [f"Average Fee: {avg_fee_for_category:.2f} sat/vB"]
            for day_name, hours_str in day_patterns.items():
                category_descriptions.append(f"  - {day_name}: {hours_str}")

            patterns_summary[category] = category_descriptions
        return patterns_summary


# --- Example Usage (demonstrates how to call the class) ---
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual PostgreSQL database configuration
    # In a production environment, use environment variables or a secure configuration system
    db_config = {
        'host': 'localhost', # e.g., 'localhost' or an IP address
        'database': 'bitcoin_blockchain',
        'user': 'postgres',
        'password': 'projectX',
        'port': 5432 # Default PostgreSQL port
    }

    # Initialize the analyzer. It will look for models in the current directory.
    # Set n_clusters to 3 for 'Low', 'Medium', 'High' categories.
    analyzer = FeePatternAnalyzer(db_config, data_interval='6 months', n_clusters=3)

    # --- Scenario 1: Train the model (e.g., when your application starts for the first time, or periodically) ---
    # This will fetch data from the DB, train the model, and save the model files.
    print("\n----- Running Training Process -----")
    if analyzer.run(train_model=True):
        print("\nTraining successful. Model is ready and saved for future use.")
    else:
        print("\nTraining failed. Check database connection, credentials, and if data exists.")
        print("Exiting example as model is not ready.")
        exit() # Exit if training fails, as subsequent steps depend on a trained model

    # --- Scenario 2: Load the model and make predictions (e.g., in a production API endpoint) ---
    # In a real API, you'd typically run `analyzer.run(train_model=False)` once when the server starts.
    # Subsequent API calls would then directly use `predict_fee_category` or `get_low_fee_recommendations`.
    print("\n----- Running Prediction Process (Loading Model) -----")
    # You could re-initialize analyzer here if this was a separate process/server restart
    # analyzer_predictor = FeePatternAnalyzer(db_config) # Re-instantiate
    if analyzer.run(train_model=False): # Attempt to load the previously saved model
        print("\nModel loaded successfully. Ready to serve predictions and patterns.")

        # --- Demonstrate Broad Pattern Extraction ---
        print("\n--- Overall Historical Fee Patterns ---")
        overall_patterns = analyzer.get_overall_fee_patterns()
        for category, descriptions in overall_patterns.items():
            print(f"\n{category} Times:")
            for desc in descriptions:
                print(f"  {desc}")

        # --- Demonstrate Specific Prediction (similar to previous example) ---
        # Get current time in Aalen, then convert to UTC for consistency with DB data
        now_aalen = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))) # Aalen is CEST, UTC+2
        current_time_utc = now_aalen.astimezone(datetime.timezone.utc)

        print(f"\n\n--- Specific Time Prediction ---")
        print(f"Current UTC Time: {analyzer.day_names[(current_time_utc.weekday() + 1) % 7]} {current_time_utc.hour:02}:00 UTC")

        # Predict for current hour
        current_day_num = (current_time_utc.weekday() + 1) % 7
        current_hour = current_time_utc.hour
        predicted_category_now = analyzer.predict_fee_category(current_day_num, current_hour)
        print(f"Predicted Category for right now: {predicted_category_now}")

        # Predict for a specific future time (e.g., next Wednesday at 15:00 UTC)
        # Remember DOW: 0=Sunday, 1=Monday, ..., 6=Saturday
        future_day_num = 3 # Wednesday
        future_hour = 15  # 15:00 UTC
        print(f"\nPredicting fee category for {analyzer.day_names[future_day_num]} {str(future_hour).zfill(2)}:00 UTC:")
        predicted_category_future = analyzer.predict_fee_category(future_day_num, future_hour)
        print(f"Predicted Category: {predicted_category_future}")

        # --- Demonstrate Low Fee Recommendations ---
        print("\n--- Low Fee Recommendations for Next 24 Hours ---")
        recommendations_24h = analyzer.get_low_fee_recommendations(num_hours_ahead=24)
        if recommendations_24h:
            for rec in recommendations_24h:
                print(f"- {rec}")
        else:
            print("No low fee recommendations available for the next 24 hours based on historical patterns.")

    else:
        print("\nCould not load model. Prediction process aborted.")