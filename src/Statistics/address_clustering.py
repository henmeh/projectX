from collections import defaultdict
import sys
import json
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import fetch_whale_transactions, store_data, create_table
import sqlite3


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


    def store_clusters(self, clusters):
        """Stores address cluster in db"""
        try:
            create_table(self.path_to_db, '''CREATE TABLE IF NOT EXISTS address_clusters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        clusters TEXT)''')
        
            for cluster in clusters:
                store_data(self.path_to_db, "INSERT INTO address_clusters (clusters) VALUES (?)", (json.dumps(cluster),))
           
        except Exception as e:
            print(f"❌ Error saving clusters to database: {e}")
    

    def fetch_clusters(self):
        try:
            conn = sqlite3.connect(self.path_to_db)
            cursor = conn.cursor()

            # Fetch all clusters from the table
            cursor.execute("SELECT clusters FROM address_clusters")
            rows = cursor.fetchall()

            # Convert the JSON string back to lists
            clusters = [json.loads(row[0]) for row in rows]

            conn.close()
            return clusters

        except Exception as e:
            print(f"❌ Error fetching clusters: {e}")
            return []