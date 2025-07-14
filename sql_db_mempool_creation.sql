set default_tablespace ='';

-- Drop existing tables
DROP TABLE IF EXISTS public.mempool_fee_histogram CASCADE;
DROP TABLE IF EXISTS public.fee_predictions CASCADE;
DROP TABLE IF EXISTS public.alerted_events CASCADE;
DROP TABLE IF EXISTS public.alert_history CASCADE;

DROP TABLE IF EXISTS public.whale_transactions CASCADE;
DROP TABLE IF EXISTS public.transactions_inputs CASCADE;
DROP TABLE IF EXISTS public.transactions_outputs CASCADE;
DROP TABLE IF EXISTS public.whale_balance_history CASCADE;
DROP TABLE IF EXISTS public.whale_behavior CASCADE;
DROP TABLE IF EXISTS public.fee_pattern CASCADE;

DROP TABLE IF EXISTS mempool_value_insights CASCADE;

-- Create essential tables
CREATE TABLE mempool_fee_histogram (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    histogram TEXT,
    fast_fee REAL,
    medium_fee REAL,
    low_fee REAL
) TABLESPACE mempool;


CREATE TABLE IF NOT EXISTS fee_pattern (
    id SERIAL PRIMARY KEY,
    analysis_timestamp TIMESTAMPTZ NOT NULL,
    fee_category VARCHAR(50) NOT NULL,
    day_of_week_num INT NOT NULL,
    start_hour INT NOT NULL,
    end_hour INT NOT NULL,
    avg_fee_for_category NUMERIC(10, 2)
) TABLESPACE mempool;


-- Fee Prediction Table
CREATE TABLE fee_predictions (
    id SERIAL PRIMARY KEY, -- PostgreSQL automatically handles this
    prediction_time TIMESTAMP WITH TIME ZONE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    fast_fee NUMERIC NOT NULL,
    medium_fee NUMERIC NOT NULL,
    low_fee NUMERIC NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
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

CREATE TABLE mempool_value_insights (
    id BIGSERIAL PRIMARY KEY,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    amount_range VARCHAR(50) NOT NULL, -- e.g., '0-1 BTC', '1-10 BTC'
    total_vsize_bytes BIGINT NOT NULL,
    avg_fee_per_vbyte REAL NOT NULL,
    transaction_count INTEGER NOT NULL,
    UNIQUE (generated_at, amount_range) -- Ensures one entry per range per generation batch
) TABLESPACE mempool;
            
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