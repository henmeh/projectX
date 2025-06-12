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

-- Create essential tables
CREATE TABLE mempool_fee_histogram (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    histogram TEXT,
    fast_fee REAL,
    medium_fee REAL,
    low_fee REAL
) TABLESPACE mempool;

CREATE TABLE fee_predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
	txid TEXT NOT NULL,
	alert_type TEXT,
	amount_btc REAL,
	amount_usd REAL
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
	FOREIGN KEY(txid) REFERENCES whale_transactions(txid)) TABLESPACE mempool;
            
CREATE TABLE transactions_outputs (
	txid TEXT,
	address TEXT,
	value REAL,
	FOREIGN KEY(txid) REFERENCES whale_transactions(txid)) TABLESPACE mempool;
	
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