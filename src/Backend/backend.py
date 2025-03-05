from fastapi import FastAPI
import sqlite3
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

DB_PATH = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"

def fetch_data(query: str, params=()):
    """Helper function to fetch data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.get("/")
def read_root():
    return {"message": "Bitcoin Analytics API"}

@app.get("/whale-transactions/")
def get_whale_transactions(min_btc: float = 10.0):
    """Fetches whale transactions above a given threshold."""
    query = "SELECT id timestamp, txid, total_sent FROM mempool_transactions WHERE total_sent >= ? ORDER BY timestamp DESC"
    transactions = fetch_data(query, (min_btc,))
    return {"whale_transactions": [{"db_id": t[0], "timestamp": t[1], "txid": t[2], "total_sent": t[3]} for t in transactions]}

@app.get("/fee-histogram/")
def get_fee_histogram():
    """Fetches the latest fee histogram data."""
    query = "SELECT timestamp, histogram FROM mempool_fee_histogram ORDER BY timestamp DESC LIMIT 1"
    result = fetch_data(query)
    if result:
        timestamp, histogram = result[0]
        return {"timestamp": timestamp, "histogram": eval(histogram)}  # Convert string back to list
    return {"error": "No data available"}

@app.get("/mempool-congestion/")
def get_mempool_congestion():
    """Checks for mempool congestion by analyzing fee histogram."""
    query = "SELECT timestamp, histogram FROM mempool_fee_histogram ORDER BY timestamp DESC LIMIT 1"
    result = fetch_data(query)
    if not result:
        return {"error": "No data available"}
    
    timestamp, histogram = result[0]
    histogram_data = eval(histogram)
    
    total_vsize = sum(entry[1] for entry in histogram_data)
    congestion_status = "High" if total_vsize > 1_000_000 else "Low"

    return {"timestamp": timestamp, "congestion_status": congestion_status, "total_vsize": total_vsize}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
