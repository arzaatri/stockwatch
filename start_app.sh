#!/usr/bin/env bash
# One command to bring the whole thing up from a clean checkout.
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
  uv run stockwatch seed --n 20
fi

cleanup() {
  echo "Stopping streaming processes..."
  kill "${STREAM_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

uv run stockwatch stream &
STREAM_PID=$!

uv run stockwatch poll --interval "${SLOW_DIM_POLL_INTERVAL_SECONDS:-3600}"
