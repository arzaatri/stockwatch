"""CLI entrypoint: `stockwatch seed|watchlist-count|run-once|poll|stream`."""

import random
import subprocess
import sys

import click

from stockwatch.config import get_settings
from stockwatch.pipeline.detect_and_explain import detect_and_explain_anomalies
from stockwatch.pipeline.poll_loop import run_forever as run_poll_forever
from stockwatch.pipeline.poll_loop import run_once as run_poll_once
from stockwatch.universe.index_constituents import fetch_index_table
from stockwatch.universe.watchlist import add_ticker, watchlist_count

STREAM_MODULES = [
    "stockwatch.streaming.quote_producer",
    "stockwatch.streaming.flink_job",
    "stockwatch.streaming.stats_consumer",
]


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--n", default=20, help="Number of tickers to seed from the S&P 500.")
def seed(n: int) -> None:
    constituents = fetch_index_table("sp500")["ticker"].to_list()
    sampled = random.sample(constituents, min(n, len(constituents)))
    for ticker in sampled:
        add_ticker(ticker)
    click.echo(f"Seeded {len(sampled)} tickers. Watchlist size: {watchlist_count()}")


@cli.command(name="watchlist-count")
def watchlist_count_command() -> None:
    click.echo(str(watchlist_count()))


@cli.command(name="run-once")
def run_once_command() -> None:
    run_poll_once()
    results = detect_and_explain_anomalies()
    if not results:
        click.echo("No anomalies detected (or not enough data yet).")
        return
    for result in results:
        context = result["context"]
        explanation = result["explanation"]
        click.echo(
            f"\n=== {context.ticker} @ {context.as_of} (score={context.anomaly_score:.4f}) ==="
        )
        click.echo(
            f"cause: {explanation.likely_cause_category} (confidence: {explanation.confidence})"
        )
        click.echo(explanation.summary)


@cli.command()
@click.option(
    "--interval", default=None, type=int, help="Seconds between metadata poll cycles."
)
def poll(interval: int | None) -> None:
    settings = get_settings()
    run_poll_forever(interval or settings.slow_dim_poll_interval_seconds)


@cli.command()
def stream() -> None:
    """Launch the real-time price path: producer + PyFlink job + stats consumer."""
    processes = [
        subprocess.Popen([sys.executable, "-m", module]) for module in STREAM_MODULES
    ]
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
