#!/usr/bin/env bash
# Tears down everything start_app.sh brings up. Safe to run even if
# start_app.sh was already Ctrl+C'd (its own trap kills the stream/dashboard
# processes it started) - this also catches anything left running from a
# start_app.sh that was backgrounded or run in another terminal.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Stopping stockwatch processes..."
pkill -f "stockwatch (stream|dashboard|poll|serve|serve-inference)" 2>/dev/null || true
pkill -f "stockwatch\.streaming\." 2>/dev/null || true
pkill -f "streamlit run .*dashboard/app\.py" 2>/dev/null || true

echo "Stopping Postgres + Redpanda..."
docker compose down

# Also wipe the Postgres data volume (drops all ingested data - only needed
# after a schema change, since this project has no migration framework):
# docker compose down -v
