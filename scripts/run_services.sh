#!/usr/bin/env bash
# Start infrastructure services (Postgres + MinIO) via Docker, then the FastAPI app.
# Usage: bash scripts/run_services.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; }

# ── 1. Check Docker ──────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install it first: https://docs.docker.com/get-docker/"
    exit 1
fi

# ── 2. Start PostgreSQL ─────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q '^acrev_db$'; then
    info "PostgreSQL already running."
else
    info "Starting PostgreSQL 18.1..."
    docker run --rm -d \
      --name acrev_db \
      -e POSTGRES_USER=acrev_user \
      -e POSTGRES_PASSWORD=acrev_password \
      -e POSTGRES_DB=acrev_db \
      -e PGDATA=/var/lib/postgresql/data \
      -p 5435:5432 \
      -v acrev_pgdata:/var/lib/postgresql \
      postgres:18.1-bookworm
    info "Waiting for Postgres to accept connections..."
    sleep 3
fi

# ── 3. Start MinIO ──────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q '^acrev_minio$'; then
    info "MinIO already running."
else
    info "Starting MinIO..."
    docker run --rm -d \
      --name acrev_minio \
      -e MINIO_ROOT_USER=minioadmin \
      -e MINIO_ROOT_PASSWORD=minioadmin \
      -p 9010:9000 \
      -p 9011:9001 \
      -v minio_data:/data \
      minio/minio:latest server /data --console-address ":9001"
fi

# ── 4. Create .env if missing ───────────────────────────────────────────────
if [ ! -f .env ]; then
    warn "No .env file found — copying from .env.example"
    cp .env.example .env
    warn "Edit .env to add your GEMINI_API_KEY for LLM-powered mode."
fi

# ── 5. Install dependencies ────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv .venv
fi

info "Installing dependencies..."
.venv/bin/pip install -e ".[dev]" --quiet 2>/dev/null || .venv/bin/pip install -e ".[dev]"

# ── 6. Run migrations ──────────────────────────────────────────────────────
info "Running database migrations..."
.venv/bin/python -m alembic upgrade head 2>/dev/null || warn "Alembic migration skipped (run manually if needed)."

# ── 7. Start the app ───────────────────────────────────────────────────────
info "Starting AEREAS API on http://localhost:8000"
info "Docs:  http://localhost:8000/docs"
info "MinIO: http://localhost:9011 (minioadmin/minioadmin)"
echo ""
.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000
