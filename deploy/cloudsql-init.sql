-- deploy/cloudsql-init.sql
-- Run once against the Cloud SQL instance after it has been created by
-- deploy/gcp-setup.sh.
--
-- IMPORTANT — Cloud SQL pgvector prerequisite:
--   The Cloud SQL instance must be created with the database flag:
--     cloudsql.enable_pgvector=on
--   This is handled by gcp-setup.sh.  If you are creating the instance
--   manually in the GCP console, add the flag under:
--     Edit instance → Flags → cloudsql.enable_pgvector = on
--   Restart the instance after enabling the flag before running this script.
--
-- How to run:
--   # Via Cloud SQL Auth Proxy (recommended):
--   cloud-sql-proxy ${PROJECT_ID}:${REGION}:rca-postgres &
--   psql "host=127.0.0.1 port=5432 dbname=rca user=rca" -f deploy/cloudsql-init.sql
--
--   # Or directly via gcloud (interactive; cannot pipe stdin on all versions):
--   gcloud sql connect rca-postgres --user=rca --database=rca
--
-- The pgvector extension is available in Cloud SQL for PostgreSQL 15 and 16
-- when the cloudsql.enable_pgvector flag is enabled.  It does NOT need to be
-- installed via apt/yum — Cloud SQL ships it as a bundled extension.

-- Enable the pgvector extension.
-- Must be run as the postgres superuser or the rca user if it has been
-- granted the cloudsqlsuperuser role.
CREATE EXTENSION IF NOT EXISTS vector;

-- Restaurants table — mirrors scripts/db_init.sql exactly.
-- Changes to the schema should be applied to both files.
CREATE TABLE IF NOT EXISTS restaurants (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    cuisine     TEXT,
    price_range FLOAT,                    -- average entree price USD
    tier        SMALLINT DEFAULT 3,       -- 4=award-winning, 3=solid, 2=mixed, 1=poor
    description TEXT,
    embedding   VECTOR(384)               -- all-MiniLM-L6-v2 dimensions (384-d)
);

-- IVFFlat ANN index — build AFTER all restaurant records have been ingested.
-- Run this separately once ingest.py has finished loading data:
--   CREATE INDEX ON restaurants USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
-- The `lists` value should be roughly sqrt(row_count); tune as needed.

-- Cloud SQL note on HNSW:
--   pgvector >= 0.5.0 supports HNSW indexes, which generally outperform
--   IVFFlat for recall.  Cloud SQL ships pgvector 0.7+ for PostgreSQL 16.
--   To use HNSW instead:
--     CREATE INDEX ON restaurants USING hnsw (embedding vector_cosine_ops);
