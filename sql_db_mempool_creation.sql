set default_tablespace ='';

-- Drop existing tables
DROP TABLE IF EXISTS public.mempool_fee_histogram CASCADE;
DROP TABLE IF EXISTS public.fee_prediction CASCADE;
DROP TABLE IF EXISTS public.alerted_events CASCADE;
DROP TABLE IF EXISTS public.alert_history CASCADE;

DROP TABLE IF EXISTS public.whale_transactions CASCADE;
DROP TABLE IF EXISTS public.transactions_inputs CASCADE;
DROP TABLE IF EXISTS public.transactions_outputs CASCADE;
DROP TABLE IF EXISTS public.whale_balance_history CASCADE;
DROP TABLE IF EXISTS public.whale_behavior CASCADE;

-- Create essential tables
CREATE TABLE mempool_fee_histogram (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    histogram TEXT,
    fast_fee REAL,
    medium_fee REAL,
    low_fee REAL
) TABLESPACE mempool;

-- Fee Prediction Table
CREATE TABLE fee_prediction (
    id SERIAL PRIMARY KEY,
    --timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prediction_timestamp TIMESTAMP NOT NULL,
    model_version INTEGER NOT NULL,
    fast_fee_pred FLOAT NOT NULL,
    medium_fee_pred FLOAT NOT NULL,
    low_fee_pred FLOAT NOT NULL
    --confidence FLOAT,
    --features JSONB
) TABLESPACE mempool;

CREATE TABLE alert_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    prediction_time INTEGER,
	fast_fee_pred REAL,
    medium_fee_pred REAL,
    low_fee_pred REAL
) TABLESPACE mempool;

CREATE TABLE alerted_events (
    event_hash TEXT PRIMARY KEY,
	timestamp INTEGER
) TABLESPACE mempool;

CREATE TABLE whale_transactions (
	txid TEXT PRIMARY KEY,
	timestamp TIMESTAMP,
	size INTEGER,
	vsize INTEGER,
	weight INTEGER,
	fee_paid REAL,
	fee_per_vbyte REAL,
	total_sent REAL,
	btcusd REAL) TABLESPACE mempool;
            
CREATE TABLE transactions_inputs (
	txid TEXT,
	address TEXT,
	value REAL,
	tx_timestamp TIMESTAMP
	--FOREIGN KEY(txid) REFERENCES whale_transactions(txid)) TABLESPACE mempool;
    ) TABLESPACE mempool;
	
CREATE TABLE transactions_outputs (
	txid TEXT,
	address TEXT,
	value REAL,
	tx_timestamp TIMESTAMP
	--FOREIGN KEY(txid) REFERENCES whale_transactions(txid)) TABLESPACE mempool;
	) TABLESPACE mempool;
	
CREATE TABLE whale_balance_history (
		address TEXT,
		timestamp TIMESTAMP,
		confirmed_balance REAL,
		unconfirmed_balance REAL,
		PRIMARY KEY (address, timestamp)) TABLESPACE mempool;
		
CREATE TABLE whale_behavior (
	address TEXT PRIMARY KEY,
	behavior_pattern TEXT,
	last_updated INTEGER) TABLESPACE mempool;

-- Create Indexes
CREATE INDEX idx_fee_histogram_timestamp ON mempool_fee_histogram(timestamp);
CREATE INDEX idx_fee_prediction_timestamp ON fee_prediction(timestamp);
CREATE INDEX idx_fee_prediction_model ON fee_prediction(model_version);



ALTER TABLE transactions_inputs ADD COLUMN tx_timestamp TIMESTAMP;
ALTER TABLE transactions_outputs ADD COLUMN tx_timestamp TIMESTAMP;

-- Update existing data
UPDATE transactions_inputs 
SET tx_timestamp = (
    SELECT timestamp 
    FROM whale_transactions 
    WHERE whale_transactions.txid = transactions_inputs.txid
);

UPDATE transactions_outputs 
SET tx_timestamp = (
    SELECT timestamp 
    FROM whale_transactions 
    WHERE whale_transactions.txid = transactions_outputs.txid
);