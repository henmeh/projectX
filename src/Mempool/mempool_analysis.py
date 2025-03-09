import pandas as pd
from collections import Counter
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
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

    
    def whale_behavior_patterns(self) -> list:
        """
        Analysing Whale transactions for recurring patterns
        """
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
        top_senders = input_counts.most_common(10)  # Top 10 frequent senders

        # Count occurrences of output (receiving) addresses
        output_counts = Counter(all_outputs)
        top_receivers = output_counts.most_common(10)  # Top 10 frequent receivers

        # Identify addresses that frequently send & receive BTC
        recurring_addresses = set(input_counts.keys()) & set(output_counts.keys())

        # Analyze Whale Activity by Time
        df["hour"] = df["timestamp"].dt.hour  # Extract hour from timestamp
        whale_activity_by_hour = df["hour"].value_counts().sort_index()

        return top_senders, top_receivers, recurring_addresses, whale_activity_by_hour
        

    def detect_unusual_activity(self, threshold: int=100)-> list:
        """
        Identify unusually large transactions
        """
        unusual_activity = []
        
        for tx in self.whale_transactions:
            total_sent = tx["total_sent"]
            
            if total_sent >= threshold:
                unusual_activity.append(tx)
        
        return unusual_activity