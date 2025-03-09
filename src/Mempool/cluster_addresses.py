import json
from mempool_analysis import AddressClustering
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, store_data

# Create an AddressClustering instance
clustering = AddressClustering()

# Get address clusters
address_clusters = clustering.run_clustering()
print(address_clusters)

'''
# Print out the first 5 clusters
for i, cluster in enumerate(address_clusters[:5]):
    print(f"Cluster {i+1}: {cluster}")

# Store address clusters in the database
def store_address_clusters(clusters, db_path):
    # Assuming we have a SQLite table for clusters
    import sqlite3
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS address_clusters (
            id INTEGER PRIMARY KEY,
            addresses TEXT
        )
    """)

    for cluster in clusters:
        # Store each cluster as a string in the database
        cluster_str = json.dumps(list(cluster))  # Convert the set to a list and then string
        cursor.execute("INSERT INTO address_clusters (addresses) VALUES (?)", (cluster_str,))
    
    connection.commit()
    connection.close()

# Store the clusters in the database
#store_address_clusters(address_clusters, "/path/to/your/db.db")
'''