"""Consumes the "price_stats" topic (produced by flink_job.py) and persists
each windowed stats record into Postgres, reusing the same db/engine.py
session layer as the rest of the codebase.
"""

import json
from datetime import UTC, datetime
from typing import Any

from kafka import KafkaConsumer
from sqlalchemy.dialects.postgresql import insert as pg_insert

from stockwatch.config import get_settings
from stockwatch.db.engine import session_scope
from stockwatch.db.models import WindowedPriceStats
from stockwatch.streaming.flink_job import STATS_TOPIC


def build_consumer(bootstrap_servers: str) -> KafkaConsumer:
    return KafkaConsumer(
        STATS_TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="stockwatch-stats-consumer",
    )


def persist_stats_record(record: dict[str, Any]) -> None:
    with session_scope() as session:
        stmt = (
            pg_insert(WindowedPriceStats)
            .values(
                ticker=record["ticker"],
                window_end=datetime.fromtimestamp(
                    record["window_end_ms"] / 1000, tz=UTC
                ),
                avg_price=record["avg_price"],
                total_volume=record["total_volume"],
                volatility_estimate=record["volatility_estimate"],
                price_zscore=record["price_zscore"],
                ingested_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["ticker", "window_end"])
        )
        session.execute(stmt)


def run_forever() -> None:
    consumer = build_consumer(get_settings().kafka_bootstrap_servers)
    for message in consumer:
        persist_stats_record(message.value)


def main() -> None:
    run_forever()


if __name__ == "__main__":
    main()
