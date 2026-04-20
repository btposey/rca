-- db_init.sql — run once after postgres container is up
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS restaurants (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    cuisine     TEXT,
    price_range FLOAT,                    -- average entree price USD
    tier        SMALLINT DEFAULT 3,       -- 4=award, 3=solid, 2=mixed, 1=poor
    description TEXT,
    embedding   VECTOR(384)               -- all-MiniLM-L6-v2 dimensions
);

-- IVFFlat index for ANN search (build after ingestion)
-- CREATE INDEX ON restaurants USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
