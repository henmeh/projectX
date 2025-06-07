import psycopg2
import json
from tqdm import tqdm
from psycopg2.extras import execute_values

class BlockchainRepair:
    def __init__(self):
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
    
    def extract_utxos_from_transactions(self):
        """Recovers UTXO data from raw transaction JSON"""
        print("Starting UTXO extraction from transactions...")
        
        # Step 1: Get all transactions with their raw data
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                # Get total transaction count for progress bar
                cur.execute("SELECT COUNT(*) FROM transactions;")
                total_txs = cur.fetchone()[0]
                
                # Fetch all transactions in batches
                batch_size = 10000
                utxos_to_insert = []
                spent_outputs = []
                
                print(f"Processing {total_txs} transactions in batches of {batch_size}...")
                
                for offset in range(0, total_txs, batch_size):
                    cur.execute("""
                        SELECT txid, block_height, timestamp, raw_tx, btc_price_usd
                        FROM transactions
                        ORDER BY block_height
                        LIMIT %s OFFSET %s;
                    """, (batch_size, offset))
                    
                    batch = cur.fetchall()
                    
                    for txid, block_height, timestamp, raw_tx_json, btc_price in batch:
                        try:
                            tx = json.loads(raw_tx_json)
                            
                            # Process outputs (new UTXOs)
                            for vout_index, output in enumerate(tx.get('vout', [])):
                                value_sat = int(output['value'] * 100000000)  # BTC to satoshis
                                
                                # Extract address from output
                                address = None
                                if 'addresses' in output.get('scriptPubKey', {}):
                                    address = output['scriptPubKey']['addresses'][0]
                                elif 'address' in output.get('scriptPubKey', {}):
                                    address = output['scriptPubKey']['address']
                                
                                if address:
                                    utxos_to_insert.append((
                                        txid, 
                                        vout_index,
                                        address,
                                        value_sat,
                                        block_height,
                                        timestamp,
                                        btc_price
                                    ))
                            
                            # Process inputs (mark UTXOs as spent)
                            for vin in tx.get('vin', []):
                                if 'txid' in vin and 'vout' in vin:
                                    spent_outputs.append((vin['txid'], vin['vout'], txid))
                        
                        except json.JSONDecodeError:
                            print(f"JSON decode error for tx: {txid}")
                    
                    # Insert UTXOs in batches
                    if utxos_to_insert:
                        execute_values(
                            cur,
                            """INSERT INTO utxos 
                            (txid, vout, address, value, block_height, timestamp, btc_price_usd) 
                            VALUES %s
                            ON CONFLICT (txid, vout) DO NOTHING""",
                            utxos_to_insert
                        )
                        utxos_to_insert = []
                    
                    # Update spent status in batches
                    if spent_outputs:
                        # Create temporary table for batch update
                        cur.execute("""
                            CREATE TEMP TABLE temp_spent (
                                txid TEXT,
                                vout INTEGER,
                                spent_in_txid TEXT
                            ) ON COMMIT DROP;
                        """)
                        
                        execute_values(
                            cur,
                            "INSERT INTO temp_spent (txid, vout, spent_in_txid) VALUES %s",
                            spent_outputs
                        )
                        
                        # Update main UTXO table
                        cur.execute("""
                            UPDATE utxos u
                            SET spent = TRUE,
                                spent_in_txid = t.spent_in_txid
                            FROM temp_spent t
                            WHERE u.txid = t.txid AND u.vout = t.vout;
                        """)
                        spent_outputs = []
                    
                    conn.commit()
                    print(f"Processed batch {offset//batch_size + 1}/{(total_txs//batch_size)+1}")
        
        print("UTXO extraction completed!")
    
    def rebuild_address_balances(self):
        """Rebuilds address balances from UTXO data"""
        print("Rebuilding address balances...")
        
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                # Create temporary table for balances
                cur.execute("""
                    CREATE TEMP TABLE temp_balances (
                        address TEXT PRIMARY KEY,
                        balance BIGINT
                    );
                """)
                
                # Calculate balances from UTXOs
                cur.execute("""
                    INSERT INTO temp_balances (address, balance)
                    SELECT address, SUM(value)
                    FROM utxos
                    WHERE NOT spent
                    GROUP BY address
                    ON CONFLICT (address) DO UPDATE
                    SET balance = EXCLUDED.balance;
                """)
                
                # Update main addresses table
                cur.execute("""
                    INSERT INTO addresses (address, balance, last_seen)
                    SELECT 
                        address, 
                        balance,
                        (SELECT MAX(timestamp) FROM utxos WHERE address = tb.address)
                    FROM temp_balances tb
                    ON CONFLICT (address) DO UPDATE
                    SET balance = EXCLUDED.balance,
                        last_seen = EXCLUDED.last_seen;
                """)
                
                conn.commit()
        
        print("Address balances rebuilt!")


    def connect_db(self):
        """Establishes a connection to PostgreSQL."""
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute("SET max_stack_depth = '7680kB';")  # Double the default
        conn.commit()
        cursor.close()
        return conn
    
    
    def get_latest_stored_block(self):
        """Returns the latest block height stored in the transactions table."""
        try:
            conn = self.connect_db()
            cursor = conn.cursor()
            # Use ORDER BY + LIMIT which can use indexes more efficiently
            cursor.execute("""
                SELECT block_height FROM transactions 
                ORDER BY block_height DESC 
                LIMIT 1;
            """)
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result[0] if result else 0
        except Exception as e:
            print(f"Error fetching latest stored block: {e}")
            return 0
    

    def find_duplicate_transactions(self):
        conn = self.connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT txid, COUNT(*) as cnt
            FROM transactions
            GROUP BY txid
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT 10;
        """)
        duplicates = cur.fetchall()
        for txid, count in duplicates:
            print(f"TXID: {txid} has {count} duplicates")
        cur.close()
        conn.close()

# Run the repair process
if __name__ == "__main__":
    repair = BlockchainRepair()
    repair.find_duplicate_transactions()
    
    # Step 1: Extract UTXOs from transaction JSON
    #repair.extract_utxos_from_transactions()
    
    # Step 2: Rebuild address balances
    #repair.rebuild_address_balances()
    
    #print("Blockchain data recovery complete!")