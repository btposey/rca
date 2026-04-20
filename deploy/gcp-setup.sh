#!/usr/bin/env bash
# deploy/gcp-setup.sh — one-time GCP infrastructure setup for RCA
#
# Run this script once from a machine that has gcloud installed and is
# authenticated as a project owner or editor.
#
# Usage:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   bash deploy/gcp-setup.sh
#
# What this script creates:
#   1. Enables required GCP APIs
#   2. Artifact Registry repository for container images
#   3. Cloud SQL (PostgreSQL 16) instance with pgvector support
#   4. GCS bucket for model weights (dispatcher-llama-1b, concierge-llama-3b)
#   5. Service account for Cloud Run
#   6. Workload Identity Federation pool + provider for GitHub Actions
#
set -euo pipefail

# ── Configuration — edit these before running ────────────────────────────────
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
GAR_REPOSITORY="rca-images"          # Artifact Registry repo name
CLOUDSQL_INSTANCE="rca-postgres"     # Cloud SQL instance name
CLOUDSQL_DB="rca"                    # Database name
CLOUDSQL_USER="rca"                  # Database user
CLOUDSQL_PASSWORD="${CLOUDSQL_PASSWORD:-}"  # Set via env or will prompt below
GCS_BUCKET_MODELS="${PROJECT_ID}-rca-models"
CLOUDRUN_SA="rca-cloudrun-sa"        # Cloud Run service account name
WIF_POOL="rca-github-pool"           # Workload Identity pool name
WIF_PROVIDER="rca-github-provider"   # Workload Identity provider name
GITHUB_ORG="${GITHUB_ORG:-}"         # Your GitHub org or username
GITHUB_REPO="${GITHUB_REPO:-rca}"    # Your GitHub repo name

# ── Validation ────────────────────────────────────────────────────────────────
if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: PROJECT_ID is required. Export it before running this script."
  exit 1
fi
if [ -z "$GITHUB_ORG" ]; then
  echo "ERROR: GITHUB_ORG is required (your GitHub username or org)."
  exit 1
fi
if [ -z "$CLOUDSQL_PASSWORD" ]; then
  read -r -s -p "Enter Cloud SQL password for user '${CLOUDSQL_USER}': " CLOUDSQL_PASSWORD
  echo
fi

echo "==> Setting active project to ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

# ── 1. Enable required GCP APIs ───────────────────────────────────────────────
echo "==> Enabling required GCP APIs (this may take a minute)..."
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  secretmanager.googleapis.com

# ── 2. Artifact Registry — container image repository ────────────────────────
echo "==> Creating Artifact Registry repository: ${GAR_REPOSITORY}"
gcloud artifacts repositories create "${GAR_REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="RCA container images (rca-api, rca-vllm, rca-ui)" \
  || echo "    (already exists — skipping)"

# ── 3. Cloud SQL — PostgreSQL 16 with pgvector ───────────────────────────────
# pgvector is enabled via the database flag cloudsql.enable_pgvector=on.
# After the instance is created, connect and run deploy/cloudsql-init.sql to
# create the extension and the restaurants table.
echo "==> Creating Cloud SQL instance: ${CLOUDSQL_INSTANCE} (PostgreSQL 16)"
echo "    This step takes 5-10 minutes..."
gcloud sql instances create "${CLOUDSQL_INSTANCE}" \
  --database-version=POSTGRES_16 \
  --region="${REGION}" \
  --tier=db-g1-small \
  --storage-type=SSD \
  --storage-size=20GB \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --database-flags=cloudsql.enable_pgvector=on \
  || echo "    (already exists — skipping)"

echo "==> Creating database: ${CLOUDSQL_DB}"
gcloud sql databases create "${CLOUDSQL_DB}" \
  --instance="${CLOUDSQL_INSTANCE}" \
  || echo "    (already exists — skipping)"

echo "==> Creating database user: ${CLOUDSQL_USER}"
gcloud sql users create "${CLOUDSQL_USER}" \
  --instance="${CLOUDSQL_INSTANCE}" \
  --password="${CLOUDSQL_PASSWORD}" \
  || echo "    (already exists — skipping)"

CLOUDSQL_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${CLOUDSQL_INSTANCE}"
echo ""
echo "    Cloud SQL connection name: ${CLOUDSQL_CONNECTION_NAME}"
echo "    Run cloudsql-init.sql once the instance is ready:"
echo "      gcloud sql connect ${CLOUDSQL_INSTANCE} --user=${CLOUDSQL_USER} --database=${CLOUDSQL_DB} < deploy/cloudsql-init.sql"
echo ""

# ── 4. GCS bucket — model weights storage ─────────────────────────────────────
# Store fine-tuned model checkpoints here so vLLM can load them at startup.
# Upload models with: gsutil -m cp -r /path/to/models/* gs://${GCS_BUCKET_MODELS}/
# vLLM reads from /models inside the container; mount the bucket via GCS FUSE.
echo "==> Creating GCS bucket for model weights: gs://${GCS_BUCKET_MODELS}"
gcloud storage buckets create "gs://${GCS_BUCKET_MODELS}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  || echo "    (already exists — skipping)"

echo "==> Creating sub-folders (placeholder objects) for model layout"
echo "placeholder" | gcloud storage cp - "gs://${GCS_BUCKET_MODELS}/dispatcher-llama-1b/.keep" || true
echo "placeholder" | gcloud storage cp - "gs://${GCS_BUCKET_MODELS}/concierge-llama-3b/.keep" || true
echo ""
echo "    Upload your fine-tuned models:"
echo "      gsutil -m cp -r /path/to/dispatcher-llama-1b gs://${GCS_BUCKET_MODELS}/dispatcher-llama-1b"
echo "      gsutil -m cp -r /path/to/concierge-llama-3b  gs://${GCS_BUCKET_MODELS}/concierge-llama-3b"
echo ""

# ── 5. Service account for Cloud Run ─────────────────────────────────────────
echo "==> Creating Cloud Run service account: ${CLOUDRUN_SA}"
gcloud iam service-accounts create "${CLOUDRUN_SA}" \
  --display-name="RCA Cloud Run Service Account" \
  || echo "    (already exists — skipping)"

CLOUDRUN_SA_EMAIL="${CLOUDRUN_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant Cloud SQL client access (for Cloud SQL Auth Proxy socket)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUDRUN_SA_EMAIL}" \
  --role="roles/cloudsql.client"

# Grant GCS read access for model weights
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET_MODELS}" \
  --member="serviceAccount:${CLOUDRUN_SA_EMAIL}" \
  --role="roles/storage.objectViewer"

# Grant Secret Manager access (for future secrets like ANTHROPIC_API_KEY)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUDRUN_SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"

echo "    Cloud Run service account: ${CLOUDRUN_SA_EMAIL}"

# ── 6. Workload Identity Federation — keyless GitHub Actions auth ─────────────
# This allows GitHub Actions to authenticate as the service account without
# storing a long-lived key in GitHub Secrets.
echo "==> Creating Workload Identity pool: ${WIF_POOL}"
gcloud iam workload-identity-pools create "${WIF_POOL}" \
  --location=global \
  --display-name="RCA GitHub Actions Pool" \
  || echo "    (already exists — skipping)"

echo "==> Creating Workload Identity provider: ${WIF_PROVIDER}"
gcloud iam workload-identity-pools providers create-oidc "${WIF_PROVIDER}" \
  --location=global \
  --workload-identity-pool="${WIF_POOL}" \
  --display-name="RCA GitHub OIDC Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${GITHUB_ORG}/${GITHUB_REPO}'" \
  || echo "    (already exists — skipping)"

# Allow the GitHub Actions identity to impersonate the Cloud Run service account
WIF_POOL_ID=$(gcloud iam workload-identity-pools describe "${WIF_POOL}" \
  --location=global \
  --format="value(name)")

gcloud iam service-accounts add-iam-policy-binding "${CLOUDRUN_SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WIF_POOL_ID}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"

# Also allow the SA to deploy to Cloud Run
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUDRUN_SA_EMAIL}" \
  --role="roles/run.admin"

# Allow the SA to push/pull from Artifact Registry
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUDRUN_SA_EMAIL}" \
  --role="roles/artifactregistry.writer"

# ── 7. Print GitHub Secrets summary ───────────────────────────────────────────
WIF_PROVIDER_FULL=$(gcloud iam workload-identity-pools providers describe "${WIF_PROVIDER}" \
  --location=global \
  --workload-identity-pool="${WIF_POOL}" \
  --format="value(name)")

echo ""
echo "================================================================"
echo " Setup complete. Add the following secrets to your GitHub repo:"
echo " (Settings → Secrets and variables → Actions → New repository secret)"
echo "================================================================"
echo ""
echo "  GCP_PROJECT_ID               = ${PROJECT_ID}"
echo "  GCP_REGION                   = ${REGION}"
echo "  GAR_REPOSITORY               = ${GAR_REPOSITORY}"
echo "  WORKLOAD_IDENTITY_PROVIDER   = ${WIF_PROVIDER_FULL}"
echo "  SERVICE_ACCOUNT              = ${CLOUDRUN_SA_EMAIL}"
echo ""
echo "  Cloud SQL connection name (for reference):"
echo "    ${CLOUDSQL_CONNECTION_NAME}"
echo ""
echo "  GCS model bucket: gs://${GCS_BUCKET_MODELS}"
echo "================================================================"
