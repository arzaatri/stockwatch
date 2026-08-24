#!/usr/bin/env bash
# One command to bring the whole thing up from a clean checkout.
if [ -z "${BASH_VERSION:-}" ]; then
  # Re-exec under bash - if this was run as `sh start_app.sh` (or sourced from
  # a non-bash shell), the shebang above gets ignored and `set -o pipefail`
  # below fails since sh/dash doesn't support it.
  exec bash "$0" "$@"
fi
set -euo pipefail

command -v java >/dev/null || {
  echo "A local JDK is required for PyFlink (e.g. apt install openjdk-17-jre-headless)." >&2
  exit 1
}

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f jars/flink-sql-connector-kafka-5.0.0-2.2.jar ]; then
  ./scripts/download_flink_jars.sh
fi

docker compose up -d
uv sync

echo "Waiting for Postgres..."
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-stockwatch}" >/dev/null 2>&1; do
  sleep 1
done

if [ "$(uv run stockwatch watchlist-count)" = "0" ]; then
  uv run stockwatch seed --n 10
fi

# Cheap on every startup: per ticker, only backfills the gap since its last
# ingested tick (capped at 7 days), not a fixed 7 days every time.
uv run stockwatch backfill --max-lookback-days 7

cleanup() {
  echo "Stopping streaming/serving processes..."
  kill "${STREAM_PID:-}" "${INFERENCE_PID:-}" "${API_PID:-}" "${DASHBOARD_PID:-}" \
    2>/dev/null || true
}
trap cleanup EXIT

uv run stockwatch stream &
STREAM_PID=$!

uv run stockwatch serve-inference &
INFERENCE_PID=$!

uv run stockwatch serve &
API_PID=$!

uv run stockwatch dashboard &
DASHBOARD_PID=$!

uv run stockwatch poll --interval "${SLOW_DIM_POLL_INTERVAL_SECONDS:-3600}"
