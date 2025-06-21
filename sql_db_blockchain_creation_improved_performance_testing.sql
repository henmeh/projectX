set default_tablespace ='';

-- Drop existing tables
DROP TABLE IF EXISTS public.transactions_test CASCADE;
DROP TABLE IF EXISTS public.utxos_test CASCADE;
DROP TABLE IF EXISTS public.addresses_test CASCADE;

-- Create essential tables
CREATE TABLE transactions_test (
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

CREATE TABLE transactions_test_000000_050000 PARTITION OF transactions_test
    FOR VALUES FROM (0) TO (50000) TABLESPACE blockchain_space; 
CREATE TABLE transactions_test_050000_100000 PARTITION OF transactions_test
    FOR VALUES FROM (50000) TO (100000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_100000_150000 PARTITION OF transactions_test
    FOR VALUES FROM (100000) TO (150000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_150000_200000 PARTITION OF transactions_test
    FOR VALUES FROM (150000) TO (200000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_200000_250000 PARTITION OF transactions_test
    FOR VALUES FROM (200000) TO (250000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_250000_300000 PARTITION OF transactions_test
    FOR VALUES FROM (250000) TO (300000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_300000_350000 PARTITION OF transactions_test
    FOR VALUES FROM (300000) TO (350000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_350000_400000 PARTITION OF transactions_test
    FOR VALUES FROM (350000) TO (400000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_400000_450000 PARTITION OF transactions_test
    FOR VALUES FROM (400000) TO (450000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_450000_500000 PARTITION OF transactions_test
    FOR VALUES FROM (450000) TO (500000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_500000_550000 PARTITION OF transactions_test
    FOR VALUES FROM (500000) TO (550000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_550000_600000 PARTITION OF transactions_test
    FOR VALUES FROM (550000) TO (600000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_600000_650000 PARTITION OF transactions_test
    FOR VALUES FROM (600000) TO (650000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_650000_700000 PARTITION OF transactions_test
    FOR VALUES FROM (650000) TO (700000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_700000_750000 PARTITION OF transactions_test
    FOR VALUES FROM (700000) TO (750000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_750000_800000 PARTITION OF transactions_test
    FOR VALUES FROM (750000) TO (800000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_800000_850000 PARTITION OF transactions_test
    FOR VALUES FROM (800000) TO (850000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_850000_900000 PARTITION OF transactions_test
    FOR VALUES FROM (850000) TO (900000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_900000_950000 PARTITION OF transactions_test
    FOR VALUES FROM (900000) TO (950000) TABLESPACE blockchain_space;
CREATE TABLE transactions_test_950000_1000000 PARTITION OF transactions_test
    FOR VALUES FROM (950000) TO (1000000) TABLESPACE blockchain_space;


CREATE TABLE utxos_test (
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

CREATE TABLE utxos_test_000000_050000 PARTITION OF utxos_test
    FOR VALUES FROM (0) TO (50000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_050000_100000 PARTITION OF utxos_test
    FOR VALUES FROM (50000) TO (100000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_100000_150000 PARTITION OF utxos_test
    FOR VALUES FROM (100000) TO (150000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_150000_200000 PARTITION OF utxos_test
    FOR VALUES FROM (150000) TO (200000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_200000_250000 PARTITION OF utxos_test
    FOR VALUES FROM (200000) TO (250000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_250000_300000 PARTITION OF utxos_test
    FOR VALUES FROM (250000) TO (300000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_300000_350000 PARTITION OF utxos_test
    FOR VALUES FROM (300000) TO (350000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_350000_400000 PARTITION OF utxos_test
    FOR VALUES FROM (350000) TO (400000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_400000_450000 PARTITION OF utxos_test
    FOR VALUES FROM (400000) TO (450000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_450000_500000 PARTITION OF utxos_test
    FOR VALUES FROM (450000) TO (500000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_500000_550000 PARTITION OF utxos_test
    FOR VALUES FROM (500000) TO (550000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_550000_600000 PARTITION OF utxos_test
    FOR VALUES FROM (550000) TO (600000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_600000_650000 PARTITION OF utxos_test
    FOR VALUES FROM (600000) TO (650000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_650000_700000 PARTITION OF utxos_test
    FOR VALUES FROM (650000) TO (700000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_700000_750000 PARTITION OF utxos_test
    FOR VALUES FROM (700000) TO (750000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_750000_800000 PARTITION OF utxos_test
    FOR VALUES FROM (750000) TO (800000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_800000_850000 PARTITION OF utxos_test
    FOR VALUES FROM (800000) TO (850000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_850001_900000 PARTITION OF utxos_test
    FOR VALUES FROM (850000) TO (900000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_900000_950000 PARTITION OF utxos_test
    FOR VALUES FROM (900000) TO (950000) TABLESPACE blockchain_space_part2;
CREATE TABLE utxos_test_950000_1000000 PARTITION OF utxos_test
    FOR VALUES FROM (950000) TO (1000000) TABLESPACE blockchain_space_part2;


CREATE TABLE addresses_test (
    address TEXT PRIMARY KEY,
    balance BIGINT NOT NULL DEFAULT 0,
    first_seen INTEGER NOT NULL,
	date_first_seen TIMESTAMP NOT NULL,
	last_seen INTEGER NOT NULL,
	date_last_seen TIMESTAMP NOT NULL
) TABLESPACE blockchain_space_part2;

-- Create Indexes
CREATE INDEX idx_transactions_test_txid ON transactions_test (txid);
CREATE INDEX idx_transactions_test_block_height ON transactions_test (block_height);
CREATE INDEX idx_transactions_test_date ON transactions_test USING BRIN (date);
CREATE INDEX idx_utxos_test_address_spent ON utxos_test (address) WHERE NOT spent;
CREATE INDEX idx_utxos_test_txid_vout ON utxos_test (txid, vout);
CREATE INDEX idx_utxos_test_spent_in ON utxos_test (spent_in_txid) WHERE spent;
CREATE INDEX idx_addresses_test_address ON addresses_test (address)