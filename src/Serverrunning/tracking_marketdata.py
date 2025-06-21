import datetime
from datetime import timedelta
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Marketdata.marketdata import Marketdata
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


if __name__ == "__main__":
    marketdata = Marketdata()
    conn = marketdata.connect_db()

    for asset_name, ticker in marketdata.get_assets().items():
        logging.info(f"⏳ Checking latest date for {asset_name}...")
        latest_date = marketdata.get_latest_date(asset_name, conn)
        print(f"📅 Latest in DB: {latest_date}")

        if latest_date is None:
            logging.info(f"📦 No data found. Fetching full history for {asset_name}...")
            hist = marketdata.fetch_historical_macro_data(ticker)
        else:
            start_date = latest_date + timedelta(days=1)
            end_date = datetime.date.today()
            if start_date > end_date:
                logging.info(f"✅ {asset_name} already up-to-date.")
                continue
            logging.info(f"📈 Fetching {asset_name} from {start_date} to {end_date}...")
            hist = marketdata.fetch_live_macro_data(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if not hist.empty:
            logging.info(f"💾 Inserting {asset_name} into database...")
            marketdata.insert_macro_data(asset_name, hist, conn)
        else:
            logging.info(f"⚠️ No new data for {asset_name}.")

    conn.close()
    logging.info("✅ All market data updated.")
