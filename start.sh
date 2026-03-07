#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload --log-level info
