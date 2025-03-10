from mempool_analysis import AddressClustering

# Create an AddressClustering instance
clustering = AddressClustering()

# Get address clusters
#address_clusters = clustering.run_clustering()
#clustering.store_clusters(address_clusters)


clusters = clustering.fetch_clusters()
clustering.visualize_clusters(clusters)