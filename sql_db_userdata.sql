CREATE TABLE trading_bot_rules (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,  -- Foreign key to users table
    rules_json TEXT NOT NULL,  -- JSON string of rules
    exchange_api_key TEXT,  -- Encrypted API key
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) TABLESPACE mempool;  -- Adjust tablespace if needed

-- Add index for fast lookup
CREATE INDEX idx_trading_bot_user ON trading_bot_rules (user_id);