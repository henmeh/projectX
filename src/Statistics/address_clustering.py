import numpy as np
from collections import defaultdict
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import networkx as nx
import sqlite3
from datetime import datetime
import hashlib
import time
from functools import lru_cache
import heapq
import psycopg2
import logging
from psycopg2.extras import execute_values
from psycopg2 import sql, DatabaseError, IntegrityError
        
        


class AddressClustering:
    def __init__(self, days: int = 7, min_cluster_size: int = 2):
        self.days = days
        self.min_cluster_size = min_cluster_size
        self.graph = nx.Graph()
        self.cluster_cache = {}
        self.cluster_version = 0
        self.last_clustering_time = 0
        self.logger = logging.getLogger('whale_clustering')

        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
        
        
    def connect_db(self):
        """Establish connection with optimized settings"""
        conn = psycopg2.connect(
            **self.db_params,
            application_name="BlockchainAnalytics",
            connect_timeout=10
        )
        
        # set critical performance parameters
        with conn.cursor() as cur:
            try:
                # Stack depth solution for recursion errors
                cur.execute("SET max_stack_depth = '7680kB';")
                
                # Query optimization flags
                cur.execute("SET enable_partition_pruning = on;")
                cur.execute("SET constraint_exclusion = 'partition';")
                cur.execute("SET work_mem = '64MB';")
                
                # Transaction configuration
                cur.execute("SET idle_in_transaction_session_timeout = '5min';")
                conn.commit()
            except psycopg2.Error as e:
                print(f"Warning: Could not set session parameters: {e}")
                conn.rollback()
        
        return conn    
    
    # ----------------------
    # Core Clustering Methods
    # ----------------------
    
    def _build_transaction_graph(self, transactions: list[dict]) -> bool:
        """Build multi-relational graph with memoization and incremental updates"""
        start_time = time.time()
        self.graph = nx.Graph()

        for tx in transactions:
            input_addrs = [input[0] for input in tx["inputs"]]
            output_addrs = [output[0] for output in tx["outputs"]]
            output_values = [output[1] for output in tx["outputs"]]

            # Common input heuristic (all inputs are connected)
            for i, _ in enumerate(input_addrs):
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

        return True
    
    
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
    def _get_address_features(self, address: str) -> tuple:
        """Robust feature extraction with UTC time handling and edge cases"""
        try:
            with self.connect_db() as conn:
                with conn.cursor() as cursor:
                    # Use UTC timestamps directly in database calculations
                    cursor.execute(
                        """
                        WITH addr_activity AS (
                            SELECT tx_timestamp AS ts FROM transactions_inputs WHERE address = %s
                            UNION ALL
                            SELECT tx_timestamp FROM transactions_outputs WHERE address = %s
                        )
                        SELECT 
                            COUNT(ts) AS tx_count,
                            EXTRACT(EPOCH FROM (NOW() AT TIME ZONE 'UTC' - MIN(ts))) AS lifetime_sec,
                            EXTRACT(EPOCH FROM (NOW() AT TIME ZONE 'UTC' - MAX(ts))) AS recency_sec,
                            CASE WHEN COUNT(ts) > 1 
                                 THEN EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) 
                                 ELSE 0 END AS active_duration_sec,
                            CASE WHEN COUNT(ts) > 1 
                                 THEN EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) / (COUNT(ts) - 1)
                                 ELSE 0 END AS avg_dwell_sec
                        FROM addr_activity
                        """,
                        (address, address)
                    )
                    row = cursor.fetchone()
                    
                    if not row or row[0] is None:
                        return (0.0, 0.0, 0.0, 0.0, 0.0)
                    
                    # Convert all values to floats immediately
                    tx_count = float(row[0])
                    lifetime_sec = float(row[1]) if row[1] else 0.0
                    recency_sec = float(row[2]) if row[2] else 0.0
                    active_duration_sec = float(row[3]) if row[3] else 0.0
                    avg_dwell_sec = float(row[4]) if row[4] else 0.0
                    
                    return (tx_count, lifetime_sec, recency_sec, active_duration_sec, avg_dwell_sec)
                    
        except Exception as e:
            self.logger.error(f"Feature error for {address}: {e}", exc_info=True)
            return (0.0, 0.0, 0.0, 0.0, 0.0)


    def _extract_features(self) -> tuple[np.ndarray, list[str]]:
        """Optimized feature extraction with bulk processing and validation"""
        if not self.graph.nodes:
            return np.array([]), []
            
        addresses = list(self.graph.nodes())
        features = []
        
        # Bulk precompute graph metrics - O(n) complexity
        degrees = dict(nx.degree(self.graph))
        clustering_coeffs = nx.clustering(self.graph)
        
        # Precompute neighbor degrees in bulk
        neighbor_degrees = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                # Handle division safely
                neighbor_degrees[node] = sum(degrees[n] for n in neighbors) / len(neighbors)
            else:
                neighbor_degrees[node] = 0.0
        
        # Process all addresses in batch
        for addr in addresses:
            # Graph features - validate numeric types
            deg = float(degrees.get(addr, 0))
            cc = float(clustering_coeffs.get(addr, 0))
            and_val = float(neighbor_degrees.get(addr, 0))
            
            # Temporal features
            try:
                temporal = self._get_address_features(addr)
                # Validate feature dimensions
                if len(temporal) != 5:
                    raise ValueError(f"Invalid feature length: {len(temporal)}")
                    
                # Convert all to floats (safe handling)
                temporal = tuple(float(x) for x in temporal)
            except Exception as e:
                self.logger.error(f"Feature error for {addr}: {str(e)}")
                temporal = (0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Combine features
            feat = [deg, cc, and_val]
            feat.extend(temporal)
            features.append(feat)
        
        # Validate output dimensions
        if len(features) != len(addresses):
            self.logger.error(f"Feature count mismatch: {len(features)} vs {len(addresses)}")
        
        return np.array(features, dtype=np.float64), addresses
    
    
    def _cluster_with_ml(self, features: np.ndarray, addresses: list[str]) -> list[list[str]]:
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
    
    
    def _merge_clusters(self, clusters: list[list[str]]) -> list[list[str]]:
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
    
    
    def _generate_cluster_id(self, addresses: list[str]) -> str:
        """Deterministic cluster ID from sorted addresses"""
        sorted_addrs = sorted(addresses)
        return hashlib.sha256(','.join(sorted_addrs).encode()).hexdigest()[:16]
    
    
    def _classify_entity(self, cluster_id: str, addresses: list[str]) -> str:
        """Classify cluster using behavioral patterns with bulk PostgreSQL queries"""
        if not addresses:
            return "unknown"
        
        try:
            with self.connect_db() as conn:
                with conn.cursor() as cursor:
                    # Bulk fetch all cluster data in one query
                    cursor.execute(
                        """
                        WITH cluster_data AS (
                            SELECT 
                                addr,
                                SUM(value)::FLOAT AS total_value,
                                COUNT(DISTINCT txid)::FLOAT AS tx_count,
                                MIN(tx_timestamp) AS first_seen,
                                MAX(tx_timestamp) AS last_seen
                            FROM (
                                SELECT 
                                    inputs.address AS addr,
                                    inputs.value,
                                    inputs.txid,
                                    inputs.tx_timestamp
                                FROM transactions_inputs inputs
                                WHERE inputs.address = ANY(%s)
                                UNION ALL
                                SELECT 
                                    outputs.address AS addr,
                                    outputs.value,
                                    outputs.txid,
                                    outputs.tx_timestamp
                                FROM transactions_outputs outputs
                                WHERE outputs.address = ANY(%s)
                            ) combined
                            GROUP BY addr
                        )
                        SELECT
                            COALESCE(SUM(total_value), 0)::FLOAT AS cluster_value,
                            COALESCE(SUM(tx_count), 0)::FLOAT AS total_tx_count,
                            MIN(first_seen) AS cluster_first_seen,
                            MAX(last_seen) AS cluster_last_seen
                        FROM cluster_data
                        """,
                        (addresses, addresses)
                    )
                    
                    row = cursor.fetchone()
                    
                    if not row:
                        return "unknown"
                    
                    cluster_value, total_tx_count, first_seen, last_seen = row
                    cluster_size = len(addresses)
                    
                    # Handle null values safely
                    cluster_value = cluster_value or 0
                    total_tx_count = total_tx_count or 0
                    first_seen = first_seen or datetime.now()
                    last_seen = last_seen or datetime.now()
                    
                    # Calculate time-based metrics
                    now = datetime.now()
                    lifetime_sec = (now - first_seen).total_seconds()
                    recency_sec = (now - last_seen).total_seconds()
                    
                    lifetime_days = lifetime_sec / 86400
                    tx_frequency = total_tx_count / max(1, lifetime_days) if lifetime_days > 0 else total_tx_count
                    
                    # Calculate value-based metrics
                    avg_value = cluster_value / cluster_size if cluster_size > 0 else 0
                    value_per_day = cluster_value / max(1, lifetime_days) if lifetime_days > 0 else cluster_value

        except Exception as e:
            self.logger.error(f"Classification error for cluster {cluster_id}: {str(e)}")
            return "unknown"
        
        # Enhanced classification logic with multiple factors
        if cluster_size > 1000:
            return "exchange_entity"
        elif avg_value > 1000:  # High value threshold
            if tx_frequency < 0.1:
                return "dormant_whale"
            elif tx_frequency > 20:
                return "exchange_hot_wallet"
            elif value_per_day > 5000:
                return "whale_accumulator"
            else:
                return "institutional"
        elif tx_frequency > 10:
            if value_per_day > 1000:
                return "payment_processor"
            else:
                return "retail_active"
        elif cluster_size > 50:
            if value_per_day > 500:
                return "mining_pool"
            else:
                return "shared_custody"
        elif recency_sec > 86400 * 90:  # >90 days inactive
            return "dormant"
        else:
            return "retail"


    # ----------------------
    # Storage & History
    # ----------------------
    
    def store_clusters(self, clusters: list[list[str]]) -> bool:
        """Fixed cluster storage with FK handling and transaction debugging"""
        self.cluster_version += 1
        current_time = datetime.now()
        
        try:
            with self.connect_db() as conn:
                with conn.cursor() as cursor:
                    # DEBUG: Start transaction with explicit FK checks
                    cursor.execute("SET CONSTRAINTS ALL DEFERRED")
                    cursor.execute("BEGIN;")
                    
                    # 1. Get current mapping
                    cursor.execute("SELECT address, cluster_id FROM address_clusters")
                    prev_mapping = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # 2. Prepare metadata - VERIFY BEFORE INSERT
                    metadata_rows = []
                    for cluster in clusters:
                        cluster_id = self._generate_cluster_id(cluster)
                        entity_type = self._classify_entity(cluster_id, cluster)
                        
                        # DEBUG: Log before metadata insert
                        self.logger.debug(f"Preparing metadata: {cluster_id} ({entity_type})")
                        
                        metadata_rows.append((
                            cluster_id, entity_type, len(cluster), 
                            current_time, current_time, self.cluster_version
                        ))
                    
                    # 3. INSERT METADATA FIRST WITH VERIFICATION
                    for row in metadata_rows:
                        try:
                            cursor.execute(
                                """INSERT INTO cluster_metadata 
                                (cluster_id, entity_type, cluster_size, first_seen, last_active, version)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (cluster_id) DO NOTHING""",
                                row
                            )
                            # DEBUG: Check rowcount
                            if cursor.rowcount == 0:
                                self.logger.warning(f"Metadata conflict for {row[0]}")
                        except Exception as e:
                            self.logger.error(f"Metadata insert failed: {str(e)}", exc_info=True)
                            raise
                    
                    # 4. Prepare addresses - VERIFY METADATA EXISTS
                    address_rows = []
                    for cluster in clusters:
                        cluster_id = self._generate_cluster_id(cluster)
                        for addr in cluster:
                            address_rows.append((addr, cluster_id))
                    
                    # CRITICAL FK CHECK
                    cluster_ids = {cid for _, cid in address_rows}
                    cursor.execute(
                        "SELECT cluster_id FROM cluster_metadata WHERE cluster_id = ANY(%s)",
                        (list(cluster_ids),)
                    )
                    existing_meta = {row[0] for row in cursor.fetchall()}
                    missing = cluster_ids - existing_meta
                    
                    if missing:
                        self.logger.critical(f"Missing metadata for clusters: {missing}")
                        raise IntegrityError("Metadata not inserted for clusters")
                    
                    # 5. Insert addresses
                    execute_values(
                        cursor,
                        """INSERT INTO address_clusters (address, cluster_id)
                        VALUES %s""",
                        address_rows,
                        page_size=1000
                    )
                    
                    # 6. Commit with explicit confirmation
                    conn.commit()
                    self.logger.info(f"Successfully committed version {self.cluster_version}")
                    
                    return True
                    
        except Exception as e:
            self.logger.critical(f"STORAGE FAILURE: {str(e)}", exc_info=True)
            # Explicit rollback
            with self.connect_db() as conn:
                conn.rollback()
            self.cluster_version -= 1
            return False
                
    # ----------------------
    # Incremental Updates
    # ----------------------
    
    def update_clusters(self, new_transactions: list[dict]) -> None:
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
    
    
    def _attach_to_existing(self, new_addresses: list[str], 
                            current_clusters: dict[str, list[str]], 
                            graph: nx.Graph) -> list[list[str]]:
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
    
    def _fetch_related_transactions(self, addresses: set[str]) -> list[dict]:
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
    # Performance Optimizations
    # ----------------------
    
    @lru_cache(maxsize=1)
    def _get_current_clusters(self) -> dict[str, list[str]]:
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
    
    def run_clustering(self) -> list[list[str]]:
        """Main clustering pipeline with performance tracking"""
        start_time = time.time()

        try:
            whale_transactions = []
            with self.connect_db() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT txid FROM whale_transactions",
                        ()
                    )
                    result_ids = cursor.fetchall()
                    whale_transaction_ids = [result_id[0] for result_id in result_ids]

                    for whale_transaction_id in whale_transaction_ids:#[0:1]:
                        cursor.execute(
                        "SELECT address, value FROM transactions_inputs WHERE txid = %s",
                        (whale_transaction_id,)
                        )
                        result_inputs = cursor.fetchall()

                        cursor.execute(
                        "SELECT address, value FROM transactions_outputs WHERE txid = %s",
                        (whale_transaction_id,)
                        )
                        result_outputs = cursor.fetchall()

                        whale_transaction = {"txid": whale_transaction_id, "inputs": result_inputs, "outputs": result_outputs}
                        whale_transactions.append(whale_transaction)

            # Build transaction graph
            self._build_transaction_graph(whale_transactions)

            # Extract features
            features, addresses = self._extract_features()
            """
                Example output and explanation
                features = [[2.00000000e+00 0.00000000e+00 1.00000000e+00 4.00000000e+00
                            2.60293532e+05 2.60293532e+05 0.00000000e+00 0.00000000e+00]
                            [1.00000000e+00 0.00000000e+00 2.00000000e+00 1.00000000e+00
                            2.60293575e+05 2.60293575e+05 0.00000000e+00 0.00000000e+00]
                            [1.00000000e+00 0.00000000e+00 2.00000000e+00 3.00000000e+00
                            2.60293594e+05 1.75680280e+05 8.46133136e+04 4.23066568e+04]]
                addresses = ['bc1qf6vc30jjmgkrayazenc8kxdatqg28jd0qhcvwc', 'bc1qzcjp2w7ujyp4gjc9nu9842238797d2enzfj6wx', '3J7cUjBZxvGRCwFBz3q23zAsnhFfZrDSSU']
            
                explanation for first row corresponding to first address:
                Index	Feature	            Unit	    Description	                Whale Significance
                0	    deg	                count	    Number of connections	    Whale addresses have higher connectivity
                1	    cc	                ratio (0-1)	Clustering coefficient	    Lower for whales (less interconnected neighbors)
                2	    avg_neighbor_deg	count	    Average neighbor degree	    Higher for whales (connect to important addresses)
                3	    tx_count	        count	    Total transactions	        Higher for exchange wallets
                4	    lifetime_sec	    seconds	    Time since first appearance	Identifies new whales
                5	    recency_sec	        seconds	    Time since last appearance	Detects awakening dormant whales
                6	    active_duration_sec	seconds	    Time between first/last tx	Short for OTC, long for accumulation
                7	    avg_dwell_sec	    seconds	    Average time between txs	Short = active trader, long = cold storage
            """

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
        
        except Exception as e:
            print(e)
    
'''
    def get_cluster_evolution(self, cluster_id: str) -> list[dict]:
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
'''

if __name__ == "__main__":
    cluster = AddressClustering()
    clusters = cluster.run_clustering()