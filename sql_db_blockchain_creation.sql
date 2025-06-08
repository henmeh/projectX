-- Drop existing tables
DROP TABLE IF EXISTS public.transactions CASCADE;
DROP TABLE IF EXISTS public.utxos CASCADE;
DROP TABLE IF EXISTS public.addresses CASCADE;

-- Create essential tables
CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    txid TEXT NOT NULL,
    block_height INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    fee INTEGER,
    size INTEGER,
    weight INTEGER,
    btc_price_usd REAL,
    raw_tx JSONB NOT NULL,
	date TIMESTAMP NOT NULL
) TABLESPACE blockchain_space;

CREATE TABLE utxos (
    id BIGSERIAL PRIMARY KEY,
    txid TEXT NOT NULL,
    vout INTEGER NOT NULL,
    address TEXT,
    value BIGINT NOT NULL,
    block_height INTEGER NOT NULL,
    spent BOOLEAN NOT NULL DEFAULT FALSE,
    timestamp INTEGER NOT NULL,
    btc_price_usd REAL,
    output_type TEXT NOT NULL,
	date TIMESTAMP NOT NULL,
	spent_in_txid TEXT
) TABLESPACE blockchain_space_part2;

CREATE TABLE addresses (
    address TEXT PRIMARY KEY,
    balance BIGINT NOT NULL DEFAULT 0,
    last_seen INTEGER NOT NULL,
	date TIMESTAMP NOT NULL
) TABLESPACE blockchain_space_part2;

-- Create Indexes
CREATE INDEX CONCURRENTLY idx_transactions_date ON transactions (transaction_date);
CREATE INDEX CONCURRENTLY idx_transactions_txid ON transactions (txid);
CREATE INDEX CONCURRENTLY idx_utxos_address ON utxos (address);
CREATE INDEX CONCURRENTLY idx_utxos_spent ON utxos (spent) WHERE NOT spent;