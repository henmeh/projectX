import datetime
import hashlib
from Helper.helperfunctions import create_table, send_telegram_alert, fetch_data, store_data

class WhaleAlerts:
    ALERT_TYPES = {
        "WHALE_MOVE": 1000000,  # $1M USD
        "MINER_BRIBE": 5000000,  # $5M USD
        "EXCHANGE_FLOW": 2000000  # $2M USD
    }
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        create_table(db_path, """CREATE TABLE IF NOT EXISTS alerted_events (
            event_hash TEXT PRIMARY KEY,
            timestamp DATETIME)""")

    def get_alert_threshold(self, alert_type: str = "WHALE_MOVE") -> int:
        return self.ALERT_TYPES.get(alert_type, 1000000)

    def _generate_event_hash(self, event_data: dict) -> str:
        """Create unique hash for alert events"""
        hash_data = f"{event_data['txid']}-{event_data['sum_usd_sent']}"
        return hashlib.sha256(hash_data.encode()).hexdigest()

    def is_alerted(self, event_hash: str) -> bool:
        """Check if event has been alerted"""
        result = fetch_data(
            self.db_path,
            "SELECT 1 FROM alerted_events WHERE event_hash = ?",
            (event_hash,)
        )
        return bool(result)

    def detect_unusual_activity(self, tx: dict):
        event_hash = self._generate_event_hash(tx)
        
        if self.is_alerted(event_hash):
            return
        
        # Determine alert type
        alert_type = "WHALE_MOVE"
        if tx["fee_per_vbyte"] > 1000:  # Extremely high fee
            alert_type = "MINER_BRIBE"
        elif any("bc1q" in addr for addr in tx["tx_out_addr"]):  # Exchange pattern
            alert_type = "EXCHANGE_FLOW"
        
        # Check if meets threshold
        if tx["sum_usd_sent"] < self.get_alert_threshold(alert_type):
            return
        
        # Send alert
        self._send_alert(tx, alert_type)
        self._mark_as_alerted(event_hash)

    def _send_alert(self, tx: dict, alert_type: str):
        """Send multi-channel alerts"""
        message = self._format_alert(tx, alert_type)
        
        # Primary channel
        send_telegram_alert(message)


    def _format_alert(self, tx: dict, alert_type: str) -> str:
        """Create formatted alert message"""
        emoji = "🐳" if alert_type == "WHALE_MOVE" else "⛏️" if alert_type == "MINER_BRIBE" else "🏦"
        return (
            f"{emoji} *{alert_type.replace('_', ' ').title()} Alert!* {emoji}\n"
            f"💰 *{tx['sum_btc_sent']:.2f} BTC* (${tx['sum_usd_sent']:,.2f})\n"
            f"⛽ *Fee Rate:* {tx['fee_per_vbyte']:.1f} sat/vB\n"
            f"📥 *From:* {', '.join(tx['tx_in_addr'][:2])}...\n"
            f"📤 *To:* {', '.join(tx['tx_out_addr'][:2])}...\n"
            f"🕒 *Time:* {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"🔗 [View Transaction](https://mempool.space/tx/{tx['txid']})"
        )

    def _mark_as_alerted(self, event_hash: str):
        """Mark event as alerted"""
        store_data(
            self.db_path,
            "INSERT OR IGNORE INTO alerted_events (event_hash, timestamp) VALUES (?, CURRENT_TIMESTAMP)",
            (event_hash,)
        )