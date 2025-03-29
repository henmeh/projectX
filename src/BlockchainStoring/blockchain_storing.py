import json
import psycopg2
from psycopg2.extras import execute_values

class BlockchainStoring:

    def __init__(self, node):
        self.node = node
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "henning",  # Replace with your PostgreSQL password
            "host": "localhost",
            "port": 5432,
        }

    def connect_db(self):
        """Establishes a connection to PostgreSQL."""
        return psycopg2.connect(**self.db_params)

    def process_block(self, block_height: int):
        """Processes a block and updates the database."""
        
        # Fetch block hash
        block_hash = self.node.rpc_call("getblockhash", [block_height])["result"]
        
        # Fetch block data
        block_data = self.node.rpc_call("getblock", [block_hash, 2])["result"]

        tx_list = block_data["tx"]
        block_time = block_data["time"]

        transactions = []
        utxos = []
        spent_utxos = []
        address_changes = {}

        for tx in tx_list:
            txid = tx["txid"]

            # Collect inputs (vin) to mark UTXOs as spent
            for vin in tx["vin"]:
                if "txid" in vin and "vout" in vin:
                    spent_utxos.append((vin["txid"], vin["vout"]))

            # Collect outputs (vout) to store new UTXOs
            for vout in tx["vout"]:
                if "scriptPubKey" in vout and "addresses" in vout["scriptPubKey"]:
                    address = vout["scriptPubKey"]["addresses"][0]  # Extract address
                    value = int(vout["value"] * 100000000)  # Convert BTC to satoshis
                    utxos.append((txid, vout["n"], address, value, block_height, False))
                    address_changes[address] = address_changes.get(address, 0) + value

            # Store transaction details
            transactions.append((txid, block_height, block_time, tx.get("fee", 0), tx["size"], tx["weight"], json.dumps(tx)))

        self.store_data(transactions, utxos, spent_utxos, address_changes)
        print(f"Processed block {block_height} with {len(tx_list)} transactions.")

    def store_data(self, transactions, utxos, spent_utxos, address_changes):
        """Inserts data into PostgreSQL efficiently."""
        conn = self.connect_db()
        cursor = conn.cursor()

        # Insert Transactions
        if transactions:
            execute_values(cursor, 
                "INSERT INTO transactions (txid, block_height, timestamp, fee, size, weight, raw_tx) VALUES %s "
                "ON CONFLICT (txid) DO NOTHING", transactions)

        # Insert new UTXOs
        if utxos:
            execute_values(cursor, 
                "INSERT INTO utxos (txid, vout, address, value, block_height, spent) VALUES %s", utxos)

        # Mark UTXOs as spent
        if spent_utxos:
            query = "UPDATE utxos SET spent = TRUE WHERE (txid, vout) IN %s"
            cursor.execute(query, (tuple(spent_utxos),))

        # Update address balances
        for address, change in address_changes.items():
            cursor.execute(
                "INSERT INTO addresses (address, balance, last_seen) VALUES (%s, %s, extract(epoch from now())::int) "
                "ON CONFLICT (address) DO UPDATE SET balance = addresses.balance + excluded.balance, last_seen = excluded.last_seen",
                (address, change))

        conn.commit()
        cursor.close()
        conn.close()

    def sync_blocks(self, start_height: int, end_height: int = None):
        """Syncs blockchain data from start_height to latest block or a given end_height."""
        
        # Get the latest block height if no end_height is provided
        if end_height is None:
            end_height = self.node.rpc_call("getblockcount", [])["result"]

        print(f"Starting block sync from {start_height} to {end_height}...")

        for height in range(start_height, end_height + 1):
            try:
                self.process_block(height)
            except Exception as e:
                print(f"Error processing block {height}: {e}")

        print("Blockchain sync complete.")

    def get_address_balance(self, address: str):
        """Fetches the balance of a given address."""
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM addresses WHERE address = %s", (address,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0