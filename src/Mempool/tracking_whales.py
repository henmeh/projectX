import time
import json
import pandas as pd
from mempool_analysis import MempoolAnalysis
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, store_data


if __name__ == "__main__":
        mempool_analysis = MempoolAnalysis()

        create_table("/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", '''CREATE TABLE IF NOT EXISTS whale_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        top_senders TEXT,
                        top_receivers TEXT,
                        recurring_addresses TEXT,
                        whale_activity_by_hour TEXT)''')

        while True:
                top_senders, top_receivers, recurring_addresses, whale_activity_by_hour = mempool_analysis.whale_behavior_patterns()

                whale_activity_by_hour_list = []

                for whale in whale_activity_by_hour:
                        whale_activity_by_hour_list.append(whale)

                store_data("/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", "INSERT INTO whale_analysis (top_senders, top_receivers, recurring_addresses, whale_activity_by_hour) VALUES (?, ?, ?, ?)", (json.dumps(top_senders), json.dumps(top_receivers), json.dumps(list(recurring_addresses)), json.dumps(whale_activity_by_hour_list)))
                print(f"✅ Data stored at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(60*60*24)  # Run once a day