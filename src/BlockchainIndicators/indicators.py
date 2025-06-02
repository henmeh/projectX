import psycopg2
from psycopg2.extras import RealDictCursor

class BlockchainIndicators:

    def __init__(self):
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }


    def daily_transaction_count(self):
        """
        Fetches daily transaction count from the 'transactions' table.
        Returns a list of dicts: [{"date": "2025-05-01", "count": 12345}, ...]
        """

        query = """
                SELECT
                    to_char(to_timestamp(timestamp), 'YYYY-MM-DD') AS day,
                    COUNT(*) AS count
                FROM transactions
                GROUP BY day
                ORDER BY day;
                """

        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                # Convert Decimal to int if needed
                return [{"date": row["day"], "count": int(row["count"])} for row in results]
    

   
indicator = BlockchainIndicators()
data = indicator.daily_transaction_count()
for entry in data[:5]:
    print(entry)