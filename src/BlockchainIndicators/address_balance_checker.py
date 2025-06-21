import psycopg2
from psycopg2.extras import RealDictCursor
import time

class AddressBalanceChecker:
    def __init__(self):
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
    
    def get_current_balance(self, address):
        """Get current balance from precomputed addresses table (instant)"""
        query = """
            SELECT balance 
            FROM addresses 
            WHERE address = %s;
        """
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (address,))
                result = cur.fetchone()
                return result[0] if result else 0
    
    def get_balance_at_block(self, address, block_height):
        """
        Get balance at specific block height using UTXO state
        (Optimized with covering index)
        """
        query = """
            SELECT COALESCE(SUM(value), 0) AS balance
            FROM utxos
            WHERE address = %s
            AND block_height <= %s
            AND (spent = FALSE OR spent_in_block > %s);
        """
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (address, block_height, block_height))
                return cur.fetchone()[0]
    
    def get_balance_history(self, address):
        """
        Get daily balance history (for charts)
        Uses materialized view for performance
        """
        query = """
            SELECT 
                date::text,
                balance
            FROM address_balance_history
            WHERE address = %s
            ORDER BY date;
        """
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (address,))
                return cur.fetchall()
    
    def create_balance_history_view(self):
        """Create materialized view for balance history (run once)"""
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS address_balance_history AS
                    SELECT
                        address,
                        date_trunc('day', to_timestamp(timestamp)) AS date,
                        SUM(CASE WHEN spent THEN -value ELSE value END)
                            OVER (PARTITION BY address ORDER BY to_timestamp(timestamp)) AS balance
                    FROM (
                        SELECT 
                            timestamp, address, value, FALSE AS spent
                        FROM utxos
                        UNION ALL
                        SELECT 
                            t.timestamp, u.address, u.value, TRUE AS spent
                        FROM utxos u
                        JOIN transactions t ON u.spent_in = t.txid
                        WHERE u.spent
                    ) balance_changes
                    ORDER BY date;
                """)
                conn.commit()
                print("Created address balance history view")

# Test the implementation
if __name__ == "__main__":
    checker = AddressBalanceChecker()
    
    # Create history view (run once)
    # checker.create_balance_history_view()2
    
    test_address = "bc1qd073gqts3cmquwqh9cha39y5lrvuffjfp5zef9"  # Genesis address
    
    # Test current balance
    start = time.time()
    balance = checker.get_current_balance(test_address)
    print(f"Current balance: {balance/100000000:.8f} BTC")
    print(f"Current balance lookup took: {time.time()-start:.4f} seconds")
    
    # Test historical balance
    #start = time.time()
    #block_500k_balance = checker.get_balance_at_block(test_address, 500000)
    #print(f"Balance at block 500,000: {block_500k_balance/100000000:.8f} BTC")
    #print(f"Historical balance lookup took: {time.time()-start:.4f} seconds")
    
    # Test balance history
    #start = time.time()
    #history = checker.get_balance_history(test_address)[:5]  # First 5 entries
    #print("Sample balance history:")
    #for entry in history:
    #    print(f"{entry['date']}: {entry['balance']/100000000:.8f} BTC")
    #print(f"Balance history lookup took: {time.time()-start:.4f} seconds")