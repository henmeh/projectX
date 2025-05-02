import yfinance as yf
import psycopg2

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("macro_data.log"),
        logging.StreamHandler()
    ]
)

class Marketdata:
    
    def __init__(self):
        self.db_params = {
            "host": "localhost",
            "port": 5432,
            "dbname": "macro_data_db",
            "user": "postgres",
            "password": ""
        }

        self.assets = {
            "BTC": "BTC-USD",
            "DXY": "DX-Y.NYB",
            "SPX": "^SPX",
            "VIX": "^VIX",
            "GOLD": "GC=F",
            "US10Y": "^TNX"
        }

    def connect_db(self):
        """Establishes a connection to PostgreSQL."""
        try:
            return psycopg2.connect(**self.db_params)
        except Exception as e:
            logging.error(f"❌ Failed to connect to the database: {e}")
            raise
    

    def get_assets(self):
        return self.assets
    

    def fetch_historical_macro_data(self, symbol, period="15y", interval="1d"):
        logging.info(f"📦 Fetching historical data for {symbol} ({period}, {interval})...")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        hist = hist.fillna({'Volume': None})
        return hist
    

    def fetch_live_macro_data(self, ticker, start, end):
        logging.info(f"📦 Fetching live data for {ticker} from {start} to {end}...")
        live = yf.download(ticker, start=start, end=end)
        live = live.fillna({'Volume': None})
        return live
    

    def get_latest_date(self, asset_symbol, conn):
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(date) FROM macro_data WHERE asset_symbol = %s;
                """, (asset_symbol,))
                result = cur.fetchone()
                return result[0]
        except Exception as e:
            logging.error(f"❌ Error fetching latest date for {asset_symbol}: {e}")
            return None
        

    def insert_macro_data(self, asset_symbol, hist, conn):
        try:
            with conn.cursor() as cur:
                for date, row in hist.iterrows():
                    try:
                        cur.execute("""
                            INSERT INTO macro_data (asset_symbol, date, open, high, low, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (asset_symbol, date) DO NOTHING;
                        """, (
                            asset_symbol,
                            date.date(),
                            row['Open'],
                            row['High'],
                            row['Low'],
                            row['Close'],
                            row['Volume'] if not row['Volume'] != row['Volume'] else None
                        ))
                    except Exception as e:
                        logging.warning(f"⚠️ Skipped row for {asset_symbol} on {date}: {e}")
            conn.commit()
            logging.info(f"✅ Inserted data for {asset_symbol} ({len(hist)} rows)")
        except Exception as e:
            logging.error(f"❌ Database error while inserting {asset_symbol}: {e}")