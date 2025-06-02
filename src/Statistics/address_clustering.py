from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from Helper.helperfunctions import fetch_whale_transactions, store_data, create_table

class AddressClustering:
    def __init__(self, db_path: str, days: int = 7):
        self.db_path = db_path
        self.days = days
        self.graph = defaultdict(set)
        self.address_embeddings = {}
        
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS address_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER,
            address TEXT,
            embedding BLOB)''')
        
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS cluster_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address_a TEXT,
            address_b TEXT,
            relationship_score REAL)''')

    def _generate_embeddings(self, transactions):
        """Generate AI embeddings for addresses based on transaction patterns"""
        # Create transaction "documents" for each address
        addr_docs = defaultdict(list)
        for tx in transactions:
            for addr in tx["tx_in_addr"]:
                addr_docs[addr].extend(tx["tx_out_addr"])
            for addr in tx["tx_out_addr"]:
                addr_docs[addr].extend(tx["tx_in_addr"])
        
        # Convert to string representations
        addr_docs = {addr: " ".join(set(docs)) for addr, docs in addr_docs.items()}
        
        # Generate TF-IDF embeddings
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(list(addr_docs.values()))
        
        # Store embeddings
        self.address_embeddings = {
            addr: embeddings[i].toarray()[0] 
            for i, addr in enumerate(addr_docs.keys())
        }

    def _cluster_with_ai(self):
        """Cluster addresses using embedding similarity"""
        if not self.address_embeddings:
            return []
        
        addresses = list(self.address_embeddings.keys())
        embeddings = np.array([self.address_embeddings[addr] for addr in addresses])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
        labels = clustering.labels_
        
        # Group addresses by cluster
        clusters = defaultdict(list)
        for addr, label in zip(addresses, labels):
            if label != -1:  # Ignore noise
                clusters[label].append(addr)
        
        return list(clusters.values())

    def run_clustering(self):
        """Run AI-enhanced clustering"""
        whale_transactions = fetch_whale_transactions(self.db_path, self.days)
        
        if not whale_transactions:
            return []
        
        self._generate_embeddings(whale_transactions)
        ai_clusters = self._cluster_with_ai()
        traditional_clusters = self._generate_clusters(whale_transactions)
        
        # Combine results
        all_clusters = traditional_clusters + ai_clusters
        merged_clusters = self._merge_clusters(all_clusters)
        
        self.store_clusters(merged_clusters)
        return merged_clusters

    def _merge_clusters(self, clusters):
        """Merge overlapping clusters"""
        # Implement graph-based merging
        cluster_graph = {}
        for i, cluster in enumerate(clusters):
            for addr in cluster:
                if addr not in cluster_graph:
                    cluster_graph[addr] = set()
                cluster_graph[addr].add(i)
        
        # Merge clusters with common addresses
        merged = []
        visited = set()
        for i in range(len(clusters)):
            if i in visited:
                continue
            current = set(clusters[i])
            queue = [j for addr in clusters[i] for j in cluster_graph[addr] if j != i]
            while queue:
                j = queue.pop()
                if j in visited:
                    continue
                visited.add(j)
                current.update(clusters[j])
                # Add new neighbors
                queue.extend(k for addr in clusters[j] for k in cluster_graph[addr] if k not in visited)
            merged.append(list(current))
        
        return merged