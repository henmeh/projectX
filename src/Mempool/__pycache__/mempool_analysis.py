import pandas as pd
from collections import Counter
from Helper.helperfunctions import fetch_whale_transactions

class MempoolAnalysis():

    def __init__(self, path_to_db: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", days: int =7):
        self.path_to_db = path_to_db
        self.days = days
        try:
            self.whale_transactions = fetch_whale_transactions(self.path_to_db, self.days)
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.whale_transactions = []
            
    
    def whale_behavior_patterns(self):
        # Convert to DataFrame
        df = pd.DataFrame(self.whale_transactions)

        # Ensure timestamps are in datetime format
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract all input addresses
        all_inputs = []
        for addresses in df["tx_in_addr"]:
            try:
                all_inputs.extend(addresses)
            except:
                continue

        # Extract all output addresses
        all_outputs = []
        for addresses in df["tx_out_addr"]:
            try:
                all_outputs.extend(addresses)  
            except:
                continue

        # Count occurrences of input (sending) addresses
        input_counts = Counter(all_inputs)
        top_senders = input_counts.most_common(1)  # Top 10 frequent senders

        # Count occurrences of output (receiving) addresses
        output_counts = Counter(all_outputs)
        top_receivers = output_counts.most_common(1)  # Top 10 frequent receivers

        # Identify addresses that frequently send & receive BTC
        recurring_addresses = set(input_counts.keys()) & set(output_counts.keys())

        # Analyze Whale Activity by Time
        df["hour"] = df["timestamp"].dt.hour  # Extract hour from timestamp
        whale_activity_by_hour = df["hour"].value_counts().sort_index()

        # Print Results
        print("📌 Top 10 Whale Senders:")
        for addr, count in top_senders:
            print(f"{addr}: {count} transactions")

        print("\n📌 Top 10 Whale Receivers:")
        for addr, count in top_receivers:
            print(f"{addr}: {count} transactions")

        print("\n🔁 Recurring Addresses (Sending & Receiving):")
        print(list(recurring_addresses)[:10])  # Show first 10 recurring addresses

        print("\n⏳ Whale Activity by Hour:")
        print(whale_activity_by_hour)
