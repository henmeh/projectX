from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2 import pool
import os
import json
import logging
from contextlib import contextmanager
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Bitcoin Analytics API", version="1.0.0")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("bitcoin-api")

# Load environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "bitcoin_blockchain")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "projectX")

# Validate environment variables
if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
    logger.error("Missing required database environment variables")
    raise RuntimeError("Database configuration is incomplete")

# Connection pool setup
try:
    connection_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="prefer"
    )
    logger.info("Database connection pool initialized")
except psycopg2.OperationalError as e:
    logger.critical(f"Failed to create database connection pool: {str(e)}")
    raise RuntimeError("Database connection failed") from e

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def get_db_connection():
    """Context manager for database connections using connection pool"""
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)


@contextmanager
def get_db_cursor():
    """Context manager for database cursors with automatic rollback on error"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error") from e
        finally:
            cursor.close()


def fetch_data(query: str, params=()):
    """Helper function to fetch data from PostgreSQL database"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
    except HTTPException:
        raise  # Re-raise already handled exceptions
    except Exception as e:
        logger.error(f"Fetch data failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Data retrieval error")


@app.on_event("shutdown")
def shutdown_event():
    """Close connection pool on shutdown"""
    if connection_pool:
        connection_pool.closeall()
        logger.info("Database connection pool closed")


@app.get("/")
def read_root():
    return {
        "message": "Bitcoin Analytics API",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/whale-transactions",
            "/fee-estimation",
            "/fee-prediction",
            "/fee-histogram",
            "/mempool-congestion",
            "/historical-fees",
            "/historical-predictions",
            "/historical-histograms"
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }, 503


@app.get("/whale-transactions/")
def get_whale_transactions(min_btc: float = 10.0, limit: int = 100):
    """Fetches whale transactions above a given threshold"""
    if min_btc < 0:
        raise HTTPException(status_code=400, detail="min_btc must be positive")
    
    if limit > 1000:
        limit = 1000
        
    query = """
        SELECT txid, timestamp, size, vsize, weight, fee_paid, fee_per_vbyte, total_sent 
        FROM whale_transactions 
        WHERE total_sent >= %s 
        ORDER BY timestamp DESC
        LIMIT %s
    """
    transactions = fetch_data(query, (min_btc, limit))
    return {"whale_transactions": transactions}


@app.get("/fee-estimation/")
def get_fee_estimation():
    """Fetches latest fee estimation"""
    query = """
        SELECT id, timestamp, fast_fee, medium_fee, low_fee 
        FROM mempool_fee_histogram 
        ORDER BY timestamp DESC 
        LIMIT 1
    """
    result = fetch_data(query)
    if result:
        return result[0]
    raise HTTPException(status_code=404, detail="No fee data available")


@app.get("/fee-prediction/{table_name}")
def get_fee_prediction(table_name: str = Path(..., description="The name of the prediction table (e.g., 'fee_predictions_prophet', 'fee_predictions_random_forest')")):
    """Fetches latest fee prediction"""
    query = f"""
        SELECT
            prediction_time,
            model_name,
            fast_fee,
            medium_fee,
            low_fee,
            generated_at
        FROM
            {table_name}
        WHERE
            generated_at = (SELECT MAX(generated_at) FROM {table_name})
            AND prediction_time >= NOW() AT TIME ZONE 'UTC' -- Use NOW() AT TIME ZONE 'UTC' for robust timezone handling
        ORDER BY
            prediction_time ASC, model_name ASC; -- Order for consistent display
    """
    result = fetch_data(query)
    if result:
        return result
    raise HTTPException(status_code=404, detail="No prediction data available")


@app.get("/fee-histogram/")
def get_fee_histogram():
    """Fetches the latest fee histogram data"""
    query = """
        SELECT timestamp, histogram 
        FROM mempool_fee_histogram 
        ORDER BY timestamp DESC 
        LIMIT 1
    """
    result = fetch_data(query)
    if not result:
        raise HTTPException(status_code=404, detail="No histogram data available")
    
    try:
        # Safely parse histogram JSON
        return {
            "timestamp": result[0]["timestamp"],
            "histogram": json.loads(result[0]["histogram"])
        }
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Histogram parse error: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid histogram format")


@app.get("/mempool-congestion/")
def get_mempool_congestion():
    """Checks for mempool congestion by analyzing fee histogram"""
    result = get_fee_histogram()
    
    try:
        histogram = result["histogram"]
        total_vsize = sum(entry[1] for entry in histogram)
        congestion_status = "High" if total_vsize > 5_000_000 else "Low"
        return {
            "timestamp": result["timestamp"],
            "congestion_status": congestion_status,
            "total_vsize": total_vsize
        }
    except (TypeError, IndexError) as e:
        logger.error(f"Congestion analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze congestion")


@app.get("/historical-fees/")
def get_historical_fees(hours: int = 24):
    """Fetches historical fee data"""
    if hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be positive")
    if hours > 168:  # Limit to 1 week max
        hours = 168
        
    query = """
        SELECT timestamp, fast_fee, medium_fee, low_fee 
        FROM mempool_fee_histogram 
        WHERE timestamp >= NOW() - INTERVAL %s 
        ORDER BY timestamp DESC
    """
    return {"historical_fees": fetch_data(query, (f"{hours} hours",))}


@app.get("/historical-predictions/")
def get_historical_predictions(hours: int = 24):
    """Fetches historical prediction data"""
    if hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be positive")
    if hours > 168:  # Limit to 1 week max
        hours = 168
        
    query = """
        SELECT prediction_timestamp, fast_fee_pred AS fast_fee, 
               medium_fee_pred AS medium_fee, low_fee_pred AS low_fee
        FROM fee_prediction 
        WHERE prediction_timestamp >= NOW() - INTERVAL %s 
        ORDER BY prediction_timestamp DESC
    """
    return {"historical_predictions": fetch_data(query, (f"{hours} hours",))}


@app.get("/historical-histograms/")
def get_historical_histogram(hours: int = 24):
    """Fetches historical prediction data"""
    if hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be positive")
    if hours > 168:  # Limit to 1 week max
        hours = 168
        
    query = """
        SELECT timestamp, histogram
        FROM mempool_fee_histogram 
        WHERE timestamp >= NOW() - INTERVAL %s 
        ORDER BY timestamp DESC
    """
    return {"historical_histograms": fetch_data(query, (f"{hours} hours",))}


# --- Endpoint to fetch latest mempool insights for the frontend ---
@app.get('/mempool-insights')
def get_mempool_insights():
    """
    Fetches the latest aggregated mempool insights (vsize and fee distribution by value sent).
    """
    query = """
            SELECT
                amount_range,
                total_vsize_bytes,
                avg_fee_per_vbyte,
                transaction_count,
                generated_at
            FROM mempool_value_insights
            WHERE
                generated_at = (SELECT MAX(generated_at) FROM mempool_value_insights)
            ORDER BY
                CASE amount_range
                    WHEN '0-1 BTC' THEN 1
                    WHEN '1-10 BTC' THEN 2
                    WHEN '10-50 BTC' THEN 3
                    WHEN '50-100 BTC' THEN 4
                    WHEN '>100 BTC' THEN 5
                    ELSE 6
                END ASC;
                """
    result = fetch_data(query)
    if result:
        return result
    raise HTTPException(status_code=404, detail="No mempool-insights data available")


if __name__ == "__main__":
    # Run using: python this_file_name.py
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=30,
        log_config=None
    )