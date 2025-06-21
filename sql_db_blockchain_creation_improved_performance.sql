set default_tablespace ='';

-- Drop existing tables
DROP TABLE IF EXISTS public.transactions CASCADE;
DROP TABLE IF EXISTS public.utxos CASCADE;
DROP TABLE IF EXISTS public.addresses CASCADE;

-- Create essential tables
CREATE TABLE transactions (
    id BIGSERIAL,
    txid TEXT NOT NULL,
    block_height INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    fee BIGINT,
    size INTEGER,
    weight INTEGER,
    btc_price_usd NUMERIC(10,2),
    raw_tx JSONB NOT NULL,
	date TIMESTAMP NOT NULL,
	PRIMARY KEY(id, block_height)
) PARTITION BY RANGE (block_height);

CREATE TABLE transactions_000000_050000 PARTITION OF transactions
    FOR VALUES FROM (0) TO (50000) TABLESPACE blockchain_space; 
CREATE TABLE transactions_050000_100000 PARTITION OF transactions
    FOR VALUES FROM (50000) TO (100000) TABLESPACE blockchain_space;
CREATE TABLE transactions_100000_150000 PARTITION OF transactions
    FOR VALUES FROM (100000) TO (150000) TABLESPACE blockchain_space;
CREATE TABLE transactions_150000_200000 PARTITION OF transactions
    FOR VALUES FROM (150000) TO (200000) TABLESPACE blockchain_space;
CREATE TABLE transactions_200000_250000 PARTITION OF transactions
    FOR VALUES FROM (200000) TO (250000) TABLESPACE blockchain_space;
CREATE TABLE transactions_250000_300000 PARTITION OF transactions
    FOR VALUES FROM (250000) TO (300000) TABLESPACE blockchain_space;
CREATE TABLE transactions_300000_350000 PARTITION OF transactions
    FOR VALUES FROM (300000) TO (350000) TABLESPACE blockchain_space;
CREATE TABLE transactions_350000_400000 PARTITION OF transactions
    FOR VALUES FROM (350000) TO (400000) TABLESPACE blockchain_space;
CREATE TABLE transactions_400000_450000 PARTITION OF transactions
    FOR VALUES FROM (400000) TO (450000) TABLESPACE blockchain_space;
CREATE TABLE transactions_450000_500000 PARTITION OF transactions
    FOR VALUES FROM (450000) TO (500000) TABLESPACE blockchain_space;
CREATE TABLE transactions_500000_550000 PARTITION OF transactions
    FOR VALUES FROM (500000) TO (550000) TABLESPACE blockchain_space;
CREATE TABLE transactions_550000_600000 PARTITION OF transactions
    FOR VALUES FROM (550000) TO (600000) TABLESPACE blockchain_space;
CREATE TABLE transactions_600000_650000 PARTITION OF transactions
    FOR VALUES FROM (600000) TO (650000) TABLESPACE blockchain_space;
CREATE TABLE transactions_650000_700000 PARTITION OF transactions
    FOR VALUES FROM (650000) TO (700000) TABLESPACE blockchain_space;
CREATE TABLE transactions_700000_750000 PARTITION OF transactions
    FOR VALUES FROM (700000) TO (750000) TABLESPACE blockchain_space;
CREATE TABLE transactions_750000_800000 PARTITION OF transactions
    FOR VALUES FROM (750000) TO (800000) TABLESPACE blockchain_space;
CREATE TABLE transactions_800000_850000 PARTITION OF transactions
    FOR VALUES FROM (800000) TO (850000) TABLESPACE blockchain_space;
CREATE TABLE transactions_850000_900000 PARTITION OF transactions
    FOR VALUES FROM (850000) TO (900000) TABLESPACE blockchain_space;
CREATE TABLE transactions_900000_950000 PARTITION OF transactions
    FOR VALUES FROM (900000) TO (950000) TABLESPACE blockchain_space;
CREATE TABLE transactions_950000_1000000 PARTITION OF transactions
    FOR VALUES FROM (950000) TO (1000000) TABLESPACE blockchain_space;


CREATE TABLE utxos (
    id BIGSERIAL,
    txid TEXT NOT NULL,
    vout INTEGER NOT NULL,
    address TEXT,
    value BIGINT NOT NULL,
    block_height INTEGER NOT NULL,
    spent BOOLEAN NOT NULL DEFAULT FALSE,
    timestamp INTEGER NOT NULL,
    btc_price_usd NUMERIC(10,2),
    output_type TEXT NOT NULL,
	date TIMESTAMP NOT NULL,
	spent_in_txid TEXT,
	PRIMARY KEY (id, block_height)
) PARTITION BY RANGE (block_height);

CREATE TABLE utxos_000000_050000 PARTITION OF utxos
    FOR VALUES FROM (0) TO (50000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_050000_100000 PARTITION OF utxos
    FOR VALUES FROM (50000) TO (100000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_100000_150000 PARTITION OF utxos
    FOR VALUES FROM (100000) TO (150000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_150000_200000 PARTITION OF utxos
    FOR VALUES FROM (150000) TO (200000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_200000_250000 PARTITION OF utxos
    FOR VALUES FROM (200000) TO (250000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_250000_300000 PARTITION OF utxos
    FOR VALUES FROM (250000) TO (300000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_300000_350000 PARTITION OF utxos
    FOR VALUES FROM (300000) TO (350000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_350000_400000 PARTITION OF utxos
    FOR VALUES FROM (350000) TO (400000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_400000_450000 PARTITION OF utxos
    FOR VALUES FROM (400000) TO (450000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_450000_500000 PARTITION OF utxos
    FOR VALUES FROM (450000) TO (500000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_500000_550000 PARTITION OF utxos
    FOR VALUES FROM (500000) TO (550000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_550000_600000 PARTITION OF utxos
    FOR VALUES FROM (550000) TO (600000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_600000_650000 PARTITION OF utxos
    FOR VALUES FROM (600000) TO (650000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_650000_700000 PARTITION OF utxos
    FOR VALUES FROM (650000) TO (700000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_700000_750000 PARTITION OF utxos
    FOR VALUES FROM (700000) TO (750000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_750000_800000 PARTITION OF utxos
    FOR VALUES FROM (750000) TO (800000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_800000_850000 PARTITION OF utxos
    FOR VALUES FROM (800000) TO (850000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_850001_900000 PARTITION OF utxos
    FOR VALUES FROM (850000) TO (900000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_900000_950000 PARTITION OF utxos
    FOR VALUES FROM (900000) TO (950000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_950000_1000000 PARTITION OF utxos
    FOR VALUES FROM (950000) TO (1000000) TABLESPACE blockchain_space_part2;


CREATE TABLE addresses (
    address TEXT PRIMARY KEY,
    balance BIGINT NOT NULL DEFAULT 0,
    first_seen INTEGER NOT NULL,
	date_first_seen TIMESTAMP NOT NULL,
	last_seen INTEGER NOT NULL,
	date_last_seen TIMESTAMP NOT NULL
) TABLESPACE blockchain_space_part2;

-- Create Indexes
CREATE INDEX idx_transactions_txid ON transactions (txid);
CREATE INDEX idx_transactions_block_height ON transactions (block_height);
CREATE INDEX idx_transactions_date ON transactions USING BRIN (date);
CREATE INDEX idx_utxos_address_spent ON utxos (address) WHERE NOT spent;
CREATE INDEX idx_utxos_txid_vout ON utxos (txid, vout);
CREATE INDEX idx_utxos_spent_in ON utxos (spent_in_txid) WHERE spent;
CREATE INDEX idx_addresses_address ON addresses (address)