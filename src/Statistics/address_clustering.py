import numpy as np
from collections import defaultdict
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import networkx as nx
import sqlite3
from datetime import datetime, timedelta
import hashlib
import time
from functools import lru_cache
import heapq
from typing import List, Dict, Tuple, Set, Any

class AddressClustering:
    def __init__(self, db_path: str, days: int = 7, min_cluster_size: int = 2):
        self.db_path = db_path
        self.days = days
        self.min_cluster_size = min_cluster_size
        self.graph = nx.Graph()
        self.cluster_cache = {}
        self.cluster_version = 0
        self.last_clustering_time = 0
        
        # Initialize database schema
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS address_clusters (
                address TEXT PRIMARY KEY,
                cluster_id TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS cluster_metadata (
                cluster_id TEXT PRIMARY KEY,
                entity_type TEXT,
                cluster_size INTEGER,
                first_seen TIMESTAMP,
                last_active TIMESTAMP,
                version INTEGER
            )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS cluster_history (
                cluster_id TEXT,
                parent_id TEXT,
                addresses TEXT,  -- Comma-separated list
                version INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_id ON address_clusters(cluster_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_address ON address_clusters(address)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_version ON cluster_metadata(version)")
    
    # ----------------------
    # Core Clustering Methods
    # ----------------------
    
    def _build_transaction_graph(self, transactions: List[Dict]) -> None:
        """Build multi-relational graph with memoization and incremental updates"""
        start_time = time.time()
        self.graph = nx.Graph()
        
        for tx in transactions:
            input_addrs = tx["tx_in_addr"]
            output_addrs = [out["address"] for out in tx["tx_out_addr"]]
            output_values = [out["value"] for out in tx["tx_out_addr"]]
            
            # Common input heuristic (all inputs are connected)
            for i in range(len(input_addrs)):
                addr1 = input_addrs[i]
                for j in range(i+1, len(input_addrs)):
                    addr2 = input_addrs[j]
                    self._add_edge(addr1, addr2, 1.0, "common_input")
                
                # Change detection heuristic
                if len(output_addrs) > 1:
                    max_value = max(output_values)
                    max_index = output_values.index(max_value)
                    for idx, addr_out in enumerate(output_addrs):
                        if idx != max_index:
                            self._add_edge(addr1, addr_out, 0.7, "change_detection")
        
        print(f"Graph built with {len(self.graph.nodes())} nodes in {time.time()-start_time:.2f}s")
    
    def _add_edge(self, addr1: str, addr2: str, weight: float, edge_type: str) -> None:
        """Add edge with cumulative weighting and type tracking"""
        if addr1 == addr2:
            return
            
        if self.graph.has_edge(addr1, addr2):
            self.graph[addr1][addr2]["weight"] += weight
            self.graph[addr1][addr2]["types"].add(edge_type)
        else:
            self.graph.add_edge(addr1, addr2, weight=weight, types={edge_type})
    
    @lru_cache(maxsize=10000)
    def _get_address_features(self, address: str) -> Tuple[float, ...]:
        """Memoized feature extraction for addresses"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM transactions WHERE address = ?",
                (address,)
            )
            row = cursor.fetchone()
            
        if not row or row[0] is None:
            return (0, 0, 0, 0)
        
        count, first_seen, last_seen = row
        now = datetime.now()
        
        # Calculate temporal features
        lifetime = (now - datetime.fromisoformat(first_seen)).days if first_seen else 0
        recency = (now - datetime.fromisoformat(last_seen)).days if last_seen else 0
        avg_freq = count / max(1, lifetime) if lifetime > 0 else count
        
        return (count, lifetime, recency, avg_freq)
    
    def _extract_features(self) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix with graph and temporal features"""
        features = []
        addresses = list(self.graph.nodes())
        
        # Graph metrics as features
        degrees = dict(nx.degree(self.graph))
        clustering_coeffs = nx.clustering(self.graph)
        
        for addr in addresses:
            # Graph features
            feat = [
                degrees.get(addr, 0),
                clustering_coeffs.get(addr, 0),
                len(list(nx.common_neighbors(self.graph, addr, addr)))  # Number of common neighbors
            ]
            
            # Temporal features (memoized)
            temporal_feats = self._get_address_features(addr)
            feat.extend(temporal_feats)
            
            features.append(feat)
        
        return np.array(features), addresses
    
    def _cluster_with_ml(self, features: np.ndarray, addresses: List[str]) -> List[List[str]]:
        """Density-based clustering with OPTICS and feature scaling"""
        if len(addresses) < self.min_cluster_size:
            return []
            
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # OPTICS with automatic parameter detection
        clustering = OPTICS(min_samples=self.min_cluster_size, metric='euclidean', n_jobs=-1)
        clustering.fit(scaled_features)
        
        # Group results
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Only clustered points
                clusters[label].append(addresses[idx])
        
        return list(clusters.values())
    
    def _merge_clusters(self, clusters: List[List[str]]) -> List[List[str]]:
        """Hierarchical merging using Jaccard similarity"""
        # Convert to sets for efficient operations
        cluster_sets = [set(c) for c in clusters]
        merged_clusters = []
        
        while cluster_sets:
            current = cluster_sets.pop()
            merged = False
            
            # Find clusters with significant overlap
            for other in cluster_sets[:]:
                intersection = current & other
                union = current | other
                jaccard = len(intersection) / len(union) if union else 0
                
                if jaccard > 0.3:  # Merge threshold
                    current |= other
                    cluster_sets.remove(other)
                    merged = True
            
            if not merged:
                merged_clusters.append(current)
        
        return [list(c) for c in merged_clusters]
    
    def _generate_cluster_id(self, addresses: List[str]) -> str:
        """Deterministic cluster ID from sorted addresses"""
        sorted_addrs = sorted(addresses)
        return hashlib.sha256(','.join(sorted_addrs).encode()).hexdigest()[:16]
    
    def _classify_entity(self, cluster_id: str, addresses: List[str]) -> str:
        """Classify cluster using behavioral patterns"""
        total_value = 0
        tx_count = 0
        min_time = float('inf')
        max_time = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for addr in addresses:
                cursor = conn.execute(
                    "SELECT SUM(value), COUNT(*), MIN(timestamp), MAX(timestamp) FROM transactions WHERE address = ?",
                    (addr,)
                )
                row = cursor.fetchone()
                
                if row and row[0] is not None:
                    addr_value, count, first_seen, last_seen = row
                    total_value += addr_value
                    tx_count += count
                    
                    if first_seen:
                        ts = datetime.fromisoformat(first_seen).timestamp()
                        min_time = min(min_time, ts)
                    if last_seen:
                        ts = datetime.fromisoformat(last_seen).timestamp()
                        max_time = max(max_time, ts)
        
        # Calculate metrics
        cluster_size = len(addresses)
        avg_value = total_value / cluster_size if cluster_size > 0 else 0
        lifetime_days = (max_time - min_time) / 86400 if min_time < float('inf') and max_time > 0 else 0
        tx_frequency = tx_count / max(1, lifetime_days) if lifetime_days > 0 else tx_count
        
        # Classification logic
        if avg_value > 1000:  # High value threshold
            if tx_frequency < 0.5:  # Low frequency
                return "whale"
            elif tx_frequency > 10:  # High frequency
                return "exchange_hot_wallet"
            else:
                return "institution"
        elif tx_frequency > 5:  # High frequency
            return "payment_processor"
        elif cluster_size > 50:  # Large cluster
            return "mining_pool"
        else:
            return "retail"
    
    # ----------------------
    # Storage & History
    # ----------------------
    
    def store_clusters(self, clusters: List[List[str]]) -> None:
        """Store clusters with versioning and history tracking"""
        self.cluster_version += 1
        current_time = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            
            # Get previous cluster mapping for history tracking
            prev_mapping = {}
            cursor = conn.execute("SELECT address, cluster_id FROM address_clusters")
            for row in cursor.fetchall():
                prev_mapping[row[0]] = row[1]
            
            # Clear previous clusters
            conn.execute("DELETE FROM address_clusters")
            
            # Insert new clusters
            for cluster in clusters:
                cluster_id = self._generate_cluster_id(cluster)
                entity_type = self._classify_entity(cluster_id, cluster)
                
                # Track cluster changes
                changed_addresses = []
                for addr in cluster:
                    prev_cid = prev_mapping.get(addr)
                    if prev_cid != cluster_id:
                        changed_addresses.append(addr)
                
                # Record history if significant change
                if changed_addresses:
                    parent_id = prev_mapping.get(changed_addresses[0], "")
                    conn.execute(
                        "INSERT INTO cluster_history (cluster_id, parent_id, addresses, version) VALUES (?, ?, ?, ?)",
                        (cluster_id, parent_id, ",".join(changed_addresses), self.cluster_version)
                    )
                
                # Insert addresses
                addr_data = [(addr, cluster_id) for addr in cluster]
                conn.executemany(
                    "INSERT INTO address_clusters (address, cluster_id) VALUES (?, ?)",
                    addr_data
                )
                
                # Insert/update metadata
                conn.execute(
                    """INSERT OR REPLACE INTO cluster_metadata 
                    (cluster_id, entity_type, cluster_size, first_seen, last_active, version)
                    VALUES (?, ?, ?, COALESCE((SELECT first_seen FROM cluster_metadata WHERE cluster_id = ?), ?), ?, ?)
                    """,
                    (cluster_id, entity_type, len(cluster), cluster_id, current_time, current_time, self.cluster_version)
                )
            
            conn.commit()
        
        # Update cache
        self.cluster_cache = {self._generate_cluster_id(c): c for c in clusters}
        self.last_clustering_time = time.time()
    
    # ----------------------
    # Incremental Updates
    # ----------------------
    
    def update_clusters(self, new_transactions: List[Dict]) -> None:
        """Incremental clustering update for new transactions"""
        if not new_transactions:
            return
            
        start_time = time.time()
        print(f"Starting incremental clustering with {len(new_transactions)} new transactions")
        
        # Get current cluster state
        current_clusters = self._get_current_clusters()
        address_to_cluster = {}
        for cid, addresses in current_clusters.items():
            for addr in addresses:
                address_to_cluster[addr] = cid
        
        # Identify new addresses
        new_addresses = set()
        for tx in new_transactions:
            input_addrs = tx["tx_in_addr"]
            output_addrs = [out["address"] for out in tx["tx_out_addr"]]
            
            for addr in input_addrs + output_addrs:
                if addr not in address_to_cluster:
                    new_addresses.add(addr)
        
        print(f"Found {len(new_addresses)} new addresses")
        
        # If no new addresses, just update timestamps
        if not new_addresses:
            print("No new addresses found in transactions")
            return
            
        # Find related transactions for new addresses
        related_txs = self._fetch_related_transactions(new_addresses)
        print(f"Fetched {len(related_txs)} related transactions")
        
        # Build partial graph only for new data
        partial_graph = nx.Graph()
        for tx in related_txs:
            input_addrs = tx["tx_in_addr"]
            output_addrs = [out["address"] for out in tx["tx_out_addr"]]
            output_values = [out["value"] for out in tx["tx_out_addr"]]
            
            # Common input heuristic
            for i in range(len(input_addrs)):
                addr1 = input_addrs[i]
                for j in range(i+1, len(input_addrs)):
                    addr2 = input_addrs[j]
                    if addr1 in new_addresses or addr2 in new_addresses:
                        partial_graph.add_edge(addr1, addr2, weight=1.0)
                
                # Change detection
                if len(output_addrs) > 1:
                    max_value = max(output_values)
                    max_index = output_values.index(max_value)
                    for idx, addr_out in enumerate(output_addrs):
                        if idx != max_index and (addr1 in new_addresses or addr_out in new_addresses):
                            partial_graph.add_edge(addr1, addr_out, weight=0.7)
        
        print(f"Built partial graph with {len(partial_graph.nodes())} nodes")
        
        # Extract features for new addresses
        new_features = []
        for addr in new_addresses:
            # Basic features (can be extended)
            feat = [
                partial_graph.degree(addr, weight='weight') if addr in partial_graph else 0,
                len(list(partial_graph.neighbors(addr))) if addr in partial_graph else 0,
            ]
            
            # Add temporal features
            temporal_feats = self._get_address_features(addr)
            feat.extend(temporal_feats)
            
            new_features.append(feat)
        
        # Cluster only new addresses
        new_addresses_list = list(new_addresses)
        if len(new_addresses_list) >= self.min_cluster_size:
            print("Running ML clustering for new addresses")
            new_clusters = self._cluster_with_ml(np.array(new_features), new_addresses_list)
        else:
            print("Attaching new addresses to existing clusters")
            new_clusters = self._attach_to_existing(new_addresses_list, current_clusters, partial_graph)
        
        print(f"Generated {len(new_clusters)} new clusters")
        
        # Merge new clusters with existing ones
        all_clusters = list(current_clusters.values()) + new_clusters
        merged_clusters = self._merge_clusters(all_clusters)
        
        # Filter by minimum size
        final_clusters = [c for c in merged_clusters if len(c) >= self.min_cluster_size]
        print(f"Merged into {len(final_clusters)} final clusters")
        
        # Store updated clusters
        self.store_clusters(final_clusters)
        print(f"Incremental update completed in {time.time()-start_time:.2f}s")
    
    def _attach_to_existing(self, new_addresses: List[str], 
                            current_clusters: Dict[str, List[str]], 
                            graph: nx.Graph) -> List[List[str]]:
        """Attach new addresses to existing clusters via graph connections"""
        new_clusters = []
        visited = set()
        
        for addr in new_addresses:
            if addr in visited:
                continue
                
            # Find connected clusters
            connected_clusters = set()
            cluster_group = set([addr])
            queue = [addr]
            
            while queue:
                current_addr = queue.pop(0)
                visited.add(current_addr)
                
                # Check if address belongs to existing cluster
                if current_addr in current_clusters:
                    connected_clusters.add(current_addr)
                
                # Explore neighbors
                if current_addr in graph:
                    for neighbor in graph.neighbors(current_addr):
                        if neighbor not in visited:
                            queue.append(neighbor)
                            cluster_group.add(neighbor)
            
            # Merge with existing clusters
            merged_cluster = set(cluster_group)
            for cid in connected_clusters:
                merged_cluster.update(current_clusters[cid])
            
            new_clusters.append(list(merged_cluster))
        
        return new_clusters
    
    #you need to do this basedon your db structure
    def _fetch_related_transactions(self, addresses: Set[str]) -> List[Dict]:
        """ACTUAL IMPLEMENTATION EXAMPLE"""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?'] * len(addresses))
            cursor = conn.execute(
                f"SELECT * FROM transactions WHERE address IN ({placeholders})",
                list(addresses)
            )
            # Convert to your transaction dictionary format
            return self._rows_to_transaction_dicts(cursor.fetchall())
    
    # ----------------------
    # Visualization & Queries
    # ----------------------
    
    def generate_cluster_graph(self, max_nodes: int = 500) -> Dict[str, Any]:
        """Generate visualization-ready graph data with cluster grouping"""
        if not self.graph.nodes:
            return {"nodes": [], "links": []}
            
        # Get cluster assignments
        cluster_map = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT address, cluster_id FROM address_clusters")
            for row in cursor.fetchall():
                cluster_map[row[0]] = row[1]
        
        # Sample nodes if too large
        nodes = list(self.graph.nodes())
        if len(nodes) > max_nodes:
            nodes = self._sample_important_nodes(max_nodes)
        
        # Build node list
        node_data = []
        for node in nodes:
            node_data.append({
                "id": node,
                "group": cluster_map.get(node, "unclustered"),
                "degree": self.graph.degree(node)
            })
        
        # Build link list
        link_data = []
        seen_edges = set()
        for edge in self.graph.edges():
            if edge[0] in nodes and edge[1] in nodes:
                # Avoid duplicate edges
                canonical_edge = tuple(sorted(edge))
                if canonical_edge not in seen_edges:
                    seen_edges.add(canonical_edge)
                    link_data.append({
                        "source": edge[0],
                        "target": edge[1],
                        "value": self.graph[edge[0]][edge[1]].get("weight", 1)
                    })
        
        return {
            "nodes": node_data,
            "links": link_data
        }
    
    def _sample_important_nodes(self, max_nodes: int) -> List[str]:
        """Sample important nodes using degree centrality"""
        if len(self.graph.nodes) <= max_nodes:
            return list(self.graph.nodes)
            
        degrees = nx.degree_centrality(self.graph)
        return heapq.nlargest(max_nodes, degrees, key=degrees.get)
    
    # ----------------------
    # Performance Optimizations
    # ----------------------
    
    @lru_cache(maxsize=1)
    def _get_current_clusters(self) -> Dict[str, List[str]]:
        """Cached cluster retrieval with version check"""
        if time.time() - self.last_clustering_time < 300:  # 5 minute cache
            return self.cluster_cache
            
        clusters = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT cluster_id, GROUP_CONCAT(address) FROM address_clusters GROUP BY cluster_id"
            )
            for row in cursor.fetchall():
                clusters[row[0]] = row[1].split(',')
        
        self.cluster_cache = clusters
        return clusters
    
    # ----------------------
    # Main Workflow
    # ----------------------
    
    def run_clustering(self, whale_transactions: List[Dict]) -> List[List[str]]:
        """Main clustering pipeline with performance tracking"""
        start_time = time.time()
        
        if not whale_transactions:
            print("No transactions provided for clustering")
            return []
        
        # Build transaction graph
        self._build_transaction_graph(whale_transactions)
        
        # Extract features
        features, addresses = self._extract_features()
        
        # Cluster using ML
        ml_clusters = self._cluster_with_ml(features, addresses)
        
        # Merge clusters
        merged_clusters = self._merge_clusters(ml_clusters)
        
        # Filter small clusters
        final_clusters = [c for c in merged_clusters if len(c) >= self.min_cluster_size]
        
        # Store results
        self.store_clusters(final_clusters)
        
        print(f"Clustering completed in {time.time()-start_time:.2f}s. "
              f"Found {len(final_clusters)} clusters.")
        
        return final_clusters
    
    def get_cluster_evolution(self, cluster_id: str) -> List[Dict]:
        """Get evolution history of a cluster"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT cluster_id, parent_id, addresses, version, timestamp 
                FROM cluster_history 
                WHERE cluster_id = ? OR parent_id = ?
                ORDER BY version""",
                (cluster_id, cluster_id)
            )
            return [{
                "cluster_id": row[0],
                "parent_id": row[1],
                "addresses": row[2].split(','),
                "version": row[3],
                "timestamp": row[4]
            } for row in cursor.fetchall()]