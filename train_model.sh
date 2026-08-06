#!/usr/bin/env bash
# Trains an IsolationForest on current data and saves it to models/, so the
# app (detect/explain/dashboard) reuses that one trained model instead of
# refitting from scratch on every call. Run this periodically (e.g. a daily
# cron) to keep it fresh - the app warns (doesn't block) if the most recent
# model is older than MODEL_STALE_AFTER_DAYS (.env).
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

docker compose up -d postgres

echo "Waiting for Postgres..."
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-stockwatch}" >/dev/null 2>&1; do
  sleep 1
done

uv run stockwatch train-model --contamination 0.05 "$@"
