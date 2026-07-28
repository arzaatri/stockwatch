"""PyFlink DataStream job: the real-time price windowing + keyed-state path.

Reads quote events from the "quotes" Kafka topic (published by quote_producer.py),
keys by ticker, aggregates a 1-minute tumbling window (avg price, total volume),
and folds each window's average into a per-ticker running mean/variance kept in
Flink's cross-window keyed ValueState - producing a z-score for how unusual this
window is relative to that ticker's own recent history. Results are published to
the "price_stats" topic for stats_consumer.py to persist into Postgres.

Runs locally via PyFlink's embedded mini-cluster - no separate Flink cluster
needed. Requires a JDK and the bundled Kafka connector jar (see jars/).
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyflink.common import Time, Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import (
    ProcessWindowFunction,
    RuntimeExecutionMode,
    StreamExecutionEnvironment,
)
from pyflink.datastream.connectors.kafka import (
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.datastream.window import TimeWindow, TumblingEventTimeWindows

from stockwatch.config import get_settings
from stockwatch.streaming.rolling_stats import TickerRunningStats, update_running_stats

QUOTES_TOPIC = "quotes"
STATS_TOPIC = "price_stats"
JARS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "jars"


class QuoteTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value: str, record_timestamp: int) -> int:
        return json.loads(value)["event_time_ms"]


class RollingStatsProcessFunction(ProcessWindowFunction):
    """Windowing: aggregates avg price / total volume within one tumbling
    window per ticker. Keyed state: folds that window's average into a
    per-ticker running mean/variance kept in `global_state()` (persists
    across window boundaries, unlike window_state()), so the z-score is
    computed against the ticker's own recent trailing behavior rather than
    only what happened in this single window.
    """

    def process(
        self, key: str, context: ProcessWindowFunction.Context, elements: Iterable[str]
    ) -> Iterable[str]:
        quotes = [json.loads(element) for element in elements]
        avg_price = sum(quote["price"] for quote in quotes) / len(quotes)
        total_volume = sum(quote["volume"] for quote in quotes)

        state_descriptor = ValueStateDescriptor("running_stats", Types.STRING())
        state = context.global_state().get_state(state_descriptor)
        raw_state = state.value()
        running_stats = (
            TickerRunningStats.model_validate_json(raw_state)
            if raw_state
            else TickerRunningStats()
        )
        new_stats, zscore = update_running_stats(running_stats, avg_price)
        state.update(new_stats.model_dump_json())

        window: TimeWindow = context.window()
        yield json.dumps(
            {
                "ticker": key,
                "window_end_ms": window.end,
                "avg_price": avg_price,
                "total_volume": total_volume,
                "volatility_estimate": new_stats.volatility,
                "price_zscore": zscore,
            }
        )


def build_kafka_source(bootstrap_servers: str) -> KafkaSource:
    return (
        KafkaSource.builder()
        .set_bootstrap_servers(bootstrap_servers)
        .set_topics(QUOTES_TOPIC)
        .set_group_id("stockwatch-flink-job")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )


def build_kafka_sink(bootstrap_servers: str) -> KafkaSink:
    serializer = (
        KafkaRecordSerializationSchema.builder()
        .set_topic(STATS_TOPIC)
        .set_value_serialization_schema(SimpleStringSchema())
        .build()
    )
    return (
        KafkaSink.builder()
        .set_bootstrap_servers(bootstrap_servers)
        .set_record_serializer(serializer)
        .build()
    )


def _extract_ticker(quote_json: str) -> str:
    return json.loads(quote_json)["ticker"]


def build_job(env: StreamExecutionEnvironment, bootstrap_servers: str) -> Any:
    watermark_strategy = (
        WatermarkStrategy.for_monotonous_timestamps().with_timestamp_assigner(
            QuoteTimestampAssigner()
        )
    )
    source = build_kafka_source(bootstrap_servers)
    quotes = env.from_source(source, watermark_strategy, "kafka-quotes")

    stats = (
        quotes.key_by(_extract_ticker, key_type=Types.STRING())
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .process(RollingStatsProcessFunction(), output_type=Types.STRING())
    )
    stats.sink_to(build_kafka_sink(bootstrap_servers))
    return stats


def main() -> None:
    settings = get_settings()

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    # The "quotes" topic has a single partition at this scale, so parallelism > 1 would leave
    # most source subtasks idle - and idle subtasks never advance their watermark, which holds
    # the merged watermark at -infinity forever and windows would never fire.
    env.set_parallelism(1)
    jar_path = next(JARS_DIR.glob("flink-sql-connector-kafka-*.jar"))
    env.add_jars(f"file://{jar_path}")

    stats = build_job(env, settings.kafka_bootstrap_servers)
    stats.print()  # visible confirmation the job is producing windowed stats

    env.execute("stockwatch-price-windowing")


if __name__ == "__main__":
    main()
