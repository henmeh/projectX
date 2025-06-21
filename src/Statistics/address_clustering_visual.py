


class AddressClusteringVisualization():



    # ----------------------
    # Visualization & Queries
    # ----------------------
    
    def generate_cluster_graph(self, max_nodes: int = 500) -> dict:
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
    
    
    def _sample_important_nodes(self, max_nodes: int) -> list[str]:
        """Sample important nodes using degree centrality"""
        if len(self.graph.nodes) <= max_nodes:
            return list(self.graph.nodes)
            
        degrees = nx.degree_centrality(self.graph)
        return heapq.nlargest(max_nodes, degrees, key=degrees.get)
