set default_tablespace ='';

-- Drop existing tables
DROP TABLE IF EXISTS public.address_clusters CASCADE;
DROP TABLE IF EXISTS public.cluster_metadata CASCADE;
DROP TABLE IF EXISTS public.cluster_history CASCADE;

-- Enable immediate constraint checks by default
SET CONSTRAINTS ALL IMMEDIATE;

-- Cluster Metadata (Core Entity)
CREATE TABLE cluster_metadata (
    cluster_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    cluster_size INTEGER NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL
) TABLESPACE mempool;

-- Address Clusters (Foreign Keys DEFERRED)
CREATE TABLE address_clusters (
    address TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES cluster_metadata(cluster_id)
        DEFERRABLE INITIALLY DEFERRED,  -- Critical for bulk operations
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
) TABLESPACE mempool;

-- History Tracking (Optimized for writes)
CREATE TABLE cluster_history (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL REFERENCES cluster_metadata(cluster_id)
        DEFERRABLE INITIALLY DEFERRED,
    parent_id TEXT,
    address TEXT NOT NULL,
    version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
) TABLESPACE mempool;

-- Performance Optimized Indexes
CREATE INDEX idx_address_cluster ON address_clusters USING HASH (cluster_id);
CREATE INDEX idx_address ON address_clusters USING HASH (address);
CREATE INDEX idx_history_version ON cluster_history (version);
CREATE INDEX idx_history_cluster ON cluster_history USING HASH (cluster_id);
CREATE INDEX idx_metadata_version ON cluster_metadata (version);

-- Special index for fast history lookups
CREATE INDEX idx_history_address_version ON cluster_history (address, version);