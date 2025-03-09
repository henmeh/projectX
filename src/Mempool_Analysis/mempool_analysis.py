from collections import Counter
import pandas as pd
from fetch_data_from_db import FetchData

# Connect to the SQLite database
DB_PATH = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"  # Change this if your DB path is different


# Example Usage
if __name__ == "__main__":

    db = FetchData()
    
    # Fetch historical transactions
    whale_transactions = db.fetch_whale_transactions(days=7)

    # Use a set to store unique transactions and avoid duplicate mempool entries
    unique_transactions = {tx["txid"]: tx for tx in whale_transactions}.values()

    # Flatten input and output addresses from unique transactions only
    all_addresses = [addr for tx in unique_transactions for addr in tx["tx_in_addr"] + tx["tx_out_addr"]]

    # Count occurrences of each address
    address_frequency = Counter(all_addresses)

    # Convert to a DataFrame for analysis
    df = pd.DataFrame(unique_transactions)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Group transactions by hour to detect timing patterns
    df["hour"] = df["timestamp"].dt.hour
    hourly_activity = df["hour"].value_counts().sort_index()

    # Compute fee statistics
    avg_fee_per_vbyte = df["fee_per_vbyte"].mean()
    max_fee_per_vbyte = df["fee_per_vbyte"].max()
    min_fee_per_vbyte = df["fee_per_vbyte"].min()

    # Find addresses appearing in multiple unique transactions
    reused_addresses = {addr: count for addr, count in address_frequency.items() if count > 1}

    # Print results
    print("Top 10 Most Used Addresses:")
    for addr, count in sorted(reused_addresses.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{addr}: {count} distinct transactions")

    print("\nWhale Transaction Activity by Hour:")
    print(hourly_activity)

    print("\nFee Analysis:")
    print(f"Avg Fee per vByte: {avg_fee_per_vbyte:.2f} sat/vB")
    print(f"Max Fee per vByte: {max_fee_per_vbyte:.2f} sat/vB")
    print(f"Min Fee per vByte: {min_fee_per_vbyte:.2f} sat/vB")