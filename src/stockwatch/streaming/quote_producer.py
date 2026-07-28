"""Polls yfinance for live quotes and (a) appends each to the raw_price_ticks
CDC log directly, and (b) publishes it to the "quotes" Kafka topic for the
PyFlink windowing job. One yfinance pull per ticker per cycle serves both.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kafka import KafkaProducer

from stockwatch.config import get_settings
from stockwatch.ingestion.prices import append_price_tick
from stockwatch.ingestion.yfinance_client import Quote, get_quote
from stockwatch.streaming.flink_job import QUOTES_TOPIC
from stockwatch.universe.watchlist import get_active_tickers

MAX_FETCH_WORKERS = 10


def build_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
    )


def quote_to_event(quote: Quote) -> dict[str, Any]:
    return {
        "ticker": quote.ticker,
        "price": quote.price,
        "volume": quote.volume or 0,
        "event_time_ms": int(quote.as_of.timestamp() * 1000),
    }


def fetch_and_publish(producer: KafkaProducer, ticker: str) -> None:
    quote = get_quote(ticker)
    if quote is None:
        return
    append_price_tick(quote)
    producer.send(QUOTES_TOPIC, key=quote.ticker, value=quote_to_event(quote))


def run_once(producer: KafkaProducer) -> None:
    tickers = get_active_tickers()
    with ThreadPoolExecutor(
        max_workers=min(MAX_FETCH_WORKERS, len(tickers) or 1)
    ) as pool:
        futures = [
            pool.submit(fetch_and_publish, producer, ticker) for ticker in tickers
        ]
        for future in futures:
            future.result()
    producer.flush()


def run_forever(interval_seconds: int) -> None:
    producer = build_producer(get_settings().kafka_bootstrap_servers)
    while True:
        run_once(producer)
        time.sleep(interval_seconds)


def main() -> None:
    run_forever(get_settings().price_poll_interval_seconds)


if __name__ == "__main__":
    main()
