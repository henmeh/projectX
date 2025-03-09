import pandas as pd
from collections import Counter
from collections import defaultdict
import sys
import json
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
    

from collections import defaultdict
import pandas as pd
import json
from Helper.helperfunctions import fetch_whale_transactions


class AddressClustering:
    def __init__(self, path_to_db: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", days: int = 7):
        self.path_to_db = path_to_db
        self.days = days
        self.address_clusters = []  # List of address clusters
        
        try:
            self.whale_transactions = fetch_whale_transactions(self.path_to_db, self.days)
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.whale_transactions = []
    
    def _generate_clusters(self, transactions):
        """
        Generates clusters of addresses that frequently interact with each other
        """
        related_addresses = defaultdict(set)  # Dictionary that tracks relationships between addresses

        # Iterate through each transaction
        for tx in transactions:
            # For each transaction, we'll add input and output addresses to each other's related set
            for input_addr in tx["tx_in_addr"]:
                for output_addr in tx["tx_out_addr"]:
                    related_addresses[input_addr].add(output_addr)
                    related_addresses[output_addr].add(input_addr)

        # Now, we can cluster addresses based on their interaction
        visited = set()  # To keep track of already visited addresses
        for address in related_addresses:
            if address not in visited:
                # Start a new cluster from this address
                cluster = self._expand_cluster(address, related_addresses, visited)
                if cluster:
                    self.address_clusters.append(cluster)

    def _expand_cluster(self, address, related_addresses, visited):
        """
        Expands a cluster by recursively adding addresses that are related to the given address.
        """
        cluster = []
        addresses_to_visit = [address]

        while addresses_to_visit:
            current_address = addresses_to_visit.pop()
            if current_address not in visited:
                visited.add(current_address)
                cluster.append(current_address)
                # Add related addresses to visit next
                addresses_to_visit.extend(related_addresses[current_address] - visited)

        # Return the cluster only if it has meaningful connections
        return cluster if len(cluster) > 1 else None  # At least two addresses interacting

    def get_clusters(self):
        """
        Retrieves the list of clusters
        """
        return self.address_clusters

    def run_clustering(self):
        """
        Run clustering analysis and get the resulting clusters
        """
        self._generate_clusters(self.whale_transactions)
        return self.get_clusters()

