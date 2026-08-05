"""CLI entrypoint: `stockwatch seed|watchlist-count|run-once|poll|stream|dashboard`."""

import random
import subprocess
import sys
from pathlib import Path

import click

from stockwatch.config import get_settings
from stockwatch.pipeline.backfill import backfill_prices
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
DASHBOARD_APP_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


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
@click.option(
    "--max-lookback-days",
    default=7,
    type=int,
    help="Cap on how far back to backfill prices. Per ticker, only fetches the "
    "gap since its last ingested tick (or this many days back if it has none).",
)
@click.option("--prices/--no-prices", default=True, help="Backfill historical prices.")
@click.option(
    "--metadata/--no-metadata",
    default=True,
    help="Backfill metadata (sector, ratings, splits, earnings, news).",
)
def backfill(max_lookback_days: int, prices: bool, metadata: bool) -> None:
    """Populate history in one shot, so you don't need quote_producer/flink_job/
    poll_loop running continuously just to accumulate enough data to detect on.
    Cheap to run on every startup - already-current tickers fetch almost nothing.
    """
    if prices:
        tick_count = backfill_prices(max_lookback_days=max_lookback_days)
        click.echo(f"Backfilled {tick_count} price ticks (max {max_lookback_days}d).")
    if metadata:
        run_poll_once()
        click.echo("Backfilled metadata for active tickers.")


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


@cli.command()
def dashboard() -> None:
    """Launch the Streamlit anomaly dashboard."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_APP_PATH)], check=True
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
