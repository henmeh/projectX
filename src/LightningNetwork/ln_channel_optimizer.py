# ln_channel_optimizer.py
from typing import Dict
from datetime import datetime, timezone, timedelta
import logging
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from FeePredictor.fee_pattern_analyzer import FeePatternAnalyzer
import psycopg2


logger = logging.getLogger(__name__)

class LNChannelOptimizer(FeePatternAnalyzer):
    """
    Optimizer for Lightning Network channel open/close timing using pre-trained fee patterns.
    Inherits from FeePatternAnalyzer to reuse prediction methods.
    """
    def __init__(self, db_config: Dict[str, any]):
        super().__init__(db_config)  # Call parent init
        self.run(train_model=False)
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
    

    def optimize_ln_channel(self, channel_size_vb: int = 140, duration_days: int = 30) -> Dict[str, any]:
        """
        Optimizes Lightning channel open/close times based on predicted low-fee windows.
        Simulates costs and savings vs. immediate action.

        Args:
            channel_size_vb (int): Estimated vB for channel tx (default 140 for simple open/close).
            duration_days (int): Channel duration in days (for close timing).

        Returns:
            Dict: {
                'optimal_open': {'time': str, 'predicted_fee': float, 'cost': float, 'savings_pct': float},
                'optimal_close': {'time': str, 'predicted_fee': float, 'cost': float, 'savings_pct': float},
                'current_fee': float,  # For baseline comparison
                'total_savings': float
            }
        Raises:
            RuntimeError: If model not ready or no low-fee windows found.
        """
        low_fee_id = next((id for id, cat in self.cluster_id_to_category.items() if cat == 'Low Fee'), None)
        if low_fee_id is None:
            raise RuntimeError("No 'Low Fee' category found.")

        now_utc = datetime.now(timezone.utc)
        low_windows_open = self.get_low_fee_recommendations(num_hours_ahead=24)  # Next day for open
        low_windows_close = self.get_low_fee_recommendations(num_hours_ahead=24 * duration_days + 24)  # Future for close

        if not low_windows_open or not low_windows_close:
            raise RuntimeError("No low-fee windows predicted.")

        # Assume current_fee from latest DB or mempool (integrate your get_mempool_feerates)
        current_fee = self._get_current_fee_estimate()

        # Select first (soonest) low window for open/close
        optimal_open = self._parse_recommendation(low_windows_open[0])
        optimal_close = self._parse_recommendation(low_windows_close[0])  # Or find best in period

        open_cost_now = current_fee * channel_size_vb
        open_cost_opt = optimal_open['fee'] * channel_size_vb
        close_cost_now = current_fee * channel_size_vb
        close_cost_opt = optimal_close['fee'] * channel_size_vb

        open_savings = ((open_cost_now - open_cost_opt) / open_cost_now * 100) if open_cost_now > 0 else 0
        close_savings = ((close_cost_now - close_cost_opt) / close_cost_now * 100) if close_cost_now > 0 else 0
        total_savings = open_cost_now + close_cost_now - open_cost_opt - close_cost_opt

        return {
            'optimal_open': {'time': optimal_open['time'], 'predicted_fee': optimal_open['fee'], 'cost': open_cost_opt, 'savings_pct': open_savings},
            'optimal_close': {'time': optimal_close['time'], 'predicted_fee': optimal_close['fee'], 'cost': close_cost_opt, 'savings_pct': close_savings},
            'current_fee': current_fee,
            'total_savings': total_savings
        }


    def _get_current_fee_estimate(self) -> float:
        # Integrate your get_mempool_feerates; fallback to avg low fee
        try:
            with self.connect_db() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT low_fee "
                            "FROM mempool_fee_histogram "
                            "ORDER BY timestamp DESC LIMIT 1"
                        )
                        last_data = cursor.fetchone()
            return last_data[0]  # Use 'low' for conservative estimate
        except:
            logger.warning("Failed to get current fee; using historical low avg.")
            return self.df_with_categories[self.df_with_categories['fee_category'] == 'Low Fee']['avg_fee'].mean() or 1.0


    def _parse_recommendation(self, rec_str: str) -> Dict[str, any]:
        # Parse "Wednesday 14:00 UTC (Avg Fee: 3.5 sat/vB)" to dict
        parts = rec_str.split(' (Avg Fee: ')
        time_part = parts[0].replace(' UTC', '')
        fee_part = float(parts[1].split(' ')[0]) if len(parts) > 1 else 0.0
        return {'time': time_part, 'fee': fee_part}
        
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual PostgreSQL database configuration.
    # In a production environment, use environment variables or a secure configuration system.
    db_config = {
        'host': 'localhost',        # e.g., 'localhost' or an IP address
        'database': 'bitcoin_blockchain',
        'user': 'postgres',
        'password': 'projectX',
        'port': 5432                # Default PostgreSQL port
    }

    # Initialize the analyzer.
    analyzer = LNChannelOptimizer(db_config)
    test = analyzer.optimize_ln_channel(140, 30)
    print(test)
    
