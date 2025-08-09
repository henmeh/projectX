# tradingbot_builder.py
from typing import Dict, List, Any, Optional
from datetime import timezone
import numpy as np
import logging
from datetime import datetime
import pandas as pd
from binance.client import Client  # pip install python-binance
import json
import psycopg2
from cryptography.fernet import Fernet  # pip install cryptography; for API key encryption
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from FeePredictor.fee_predictor_random_forest import FeePredictorRandomForest  # Import updated class

logger = logging.getLogger(__name__)

# Generate encryption key (store securely, e.g., env var)
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

class TradingBotBuilder:
    def __init__(self, rf_model_dir: str = './trained_models_random_forest/'):
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
        self.predictor = FeePredictorRandomForest('postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain', 'mempool_fee_histogram', 'fee_predictions_random_forest', model_dir=rf_model_dir)
        self.predictor.load_latest_predictions()  # Run to load/generate predictions


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
    
    
    def build_bot_rule(self, user_id: int, rules: List[Dict[str, Any]], exchange_api_key: Optional[str] = None) -> int:
        """
        Saves user bot rules to DB. Rules e.g.: [{'condition': 'fees < 5 and hodl_waves > 80', 'action': 'buy', 'amount': 100}].
        Encrypts API key if provided (premium).
        Returns rule ID.
        """
        encrypted_key = cipher.encrypt(exchange_api_key.encode()).decode() if exchange_api_key else None
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO trading_bot_rules (user_id, rules_json, exchange_api_key, created_at) VALUES (%s, %s, %s, %s) RETURNING id",
                (user_id, json.dumps(rules), encrypted_key, datetime.now(timezone.utc))
            )
            rule_id = cursor.fetchone()[0]
            conn.commit()
        return rule_id


    def backtest_bot(self, rule_id: int, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Backtests rules on historical data with mocked hodl_waves and btc_price.
        Returns performance (e.g., PNL, trades).
        """
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rules_json FROM trading_bot_rules WHERE id = %s", (rule_id,))
            rules = json.loads(cursor.fetchone()[0])

        # Load historical fee data
        historical_df = self._load_historical_for_backtest(start_date, end_date)

        historical_df = pd.read_csv("mockdata.csv", parse_dates=['timestamp'])
        print(historical_df.head())

        if historical_df.empty:
            raise ValueError(f"No historical data found for the period {start_date} to {end_date}. Please check your database or adjust the date range.")

        trades = []
        pnl = 0
        for _, row in historical_df.iterrows():
            for rule in rules:
                if self._evaluate_condition(rule['condition'], row):
                    trade = self._simulate_trade(rule['action'], rule['amount'], row['btc_price'], row['timestamp'])
                    pnl += trade['pnl']
                    trades.append(trade)

        return {'pnl': pnl, 'trades': trades, 'performance_metrics': {'sharpe': self._calculate_sharpe(trades)}}


    def _load_historical_for_backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        # Query DB for historical fees only (timestamp, fast_fee, medium_fee, low_fee)
        query = f"""
            SELECT timestamp, fast_fee, medium_fee, low_fee
            FROM {self.predictor.historical_table_name}
            WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY timestamp
        """
        df = pd.read_sql(query, self.predictor.engine, parse_dates=['timestamp'])
        return df


    def _evaluate_condition(self, condition: str, row: pd.Series) -> bool:
        # Use latest hourly prediction for fees if available, else use row data
        fees = self.predictor.latest_predictions.get('hourly', pd.DataFrame()).loc[row['timestamp'], 'low_fee'] if row['timestamp'] in self.predictor.latest_predictions.get('hourly', pd.DataFrame()).index else row['low_fee']
        hodl = row['hodl_waves']
        # Replace and eval (use restricted globals in prod)
        cond = condition.replace('fees', str(fees < 5)).replace('hodl_waves', str(hodl))
        return eval(cond, {"__builtins__": {}}, {})


    def _simulate_trade(self, action: str, amount: float, price: float, timestamp: datetime) -> Dict:
        # Mock trade PNL with realistic 2-5% fluctuation
        fluctuation = np.random.uniform(0.02, 0.05)
        if action == 'buy':
            future_price = price * (1 + fluctuation)
            pnl = (future_price - price) * (amount / price)
        else:
            pnl = 0
        return {'pnl': pnl, 'time': timestamp, 'action': action}


    def _calculate_sharpe(self, trades: List[Dict]) -> float:
        returns = [t['pnl'] / 100 for t in trades if t['pnl'] != 0]  # Normalize to percentage
        return np.mean(returns) / np.std(returns) if returns else 0


    def execute_live_bot(self, rule_id: int, is_premium: bool = False) -> str:
        if not is_premium:
            raise ValueError("Live execution requires premium subscription.")
        
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rules_json, exchange_api_key FROM trading_bot_rules WHERE id = %s", (rule_id,))
            rules_str, encrypted_key = cursor.fetchone()
            rules = json.loads(rules_str)
            api_key = cipher.decrypt(encrypted_key.encode()).decode() if encrypted_key else None

        if not api_key:
            raise ValueError("No API key for live execution.")

        client = Client(api_key)  # Binance example
        current_data = self._get_current_data()
        for rule in rules:
            if self._evaluate_condition(rule['condition'], current_data):
                if rule['action'] == 'buy':
                    client.order_market_buy(symbol='BTCUSDT', quantity=rule['amount'] / current_data['btc_price'])
                # Log trade

        return "Trade executed"


    def _get_current_data(self) -> pd.Series:
        # Fetch current fees (from mempool), on-chain (HODL waves mock/query), BTC price (API)
        current_hour = datetime.now(timezone.utc).hour
        current_day = datetime.now(timezone.utc).weekday()
        fees = self.predictor.latest_predictions.get('hourly', pd.DataFrame()).loc[pd.Timestamp.now(tz='UTC'), 'low_fee'] if pd.Timestamp.now(tz='UTC') in self.predictor.latest_predictions.get('hourly', pd.DataFrame()).index else 0
        hodl_waves = 85.0  # Mock or query
        btc_price = 200000.0  # Mock BTC price for August 2025
        return pd.Series({'fees': fees, 'hodl_waves': hodl_waves, 'btc_price': btc_price, 'day_of_week_num': current_day, 'hour': current_hour})


# Manual Testing Block
if __name__ == "__main__":
    # Initialize
    bot_builder = TradingBotBuilder()

    # Test 1: Build a Bot Rule
    user_id = 1  # Mock user ID
    rules = [
        {'condition': 'fees < 5 and hodl_waves > 80', 'action': 'buy', 'amount': 100},
        {'condition': 'fees > 20', 'action': 'sell', 'amount': 50}
    ]
    exchange_api_key = "your_binance_api_key"  # Mock for testing; remove in prod
    rule_id = bot_builder.build_bot_rule(user_id, rules, exchange_api_key)
    print(f"Created rule ID: {rule_id}")

    # Test 2: Backtest the Bot
    start_date = "2025-07-01"
    end_date = "2025-07-31"
    backtest_result = bot_builder.backtest_bot(rule_id, start_date, end_date)
    print(f"Backtest Result: PNL={backtest_result['pnl']:.2f}, Trades={len(backtest_result['trades'])}, Sharpe={backtest_result['performance_metrics']['sharpe']:.2f}")

    """
    # Test 3: Execute Live Bot (Mocked for safety)
    try:
        execute_result = bot_builder.execute_live_bot(rule_id, is_premium=True)
        print(f"Live Execution Result: {execute_result}")
    except ValueError as e:
        print(f"Live Execution Error (expected without real API): {e}")
    """