# stockwatch

An explainable stock anomaly detection pipeline: real-time price streaming (Redpanda + PyFlink),
slow-changing metadata (sector, index membership, analyst ratings, splits, earnings, news) tracked
with CDC/SCD2 in Postgres, IsolationForest anomaly detection, SHAP explanations, and an LLM
(Gemini, via LangChain/LangGraph) that reasons about the likely cause.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- A local JDK (11 or 17), for PyFlink - e.g. `apt install openjdk-17-jre-headless`
- A free [Google AI Studio](https://aistudio.google.com/apikey) API key for Gemini (only needed
  for the actual LLM explanation call - everything else runs without it)

## Setup

```bash
cp .env.example .env   # then fill in GEMINI_API_KEY
./start_app.sh
```

`start_app.sh` downloads the Flink Kafka connector jar (first run only), brings up Postgres +
Redpanda, seeds the watchlist with 20 random S&P 500 tickers (first run only), and starts both
the real-time price stream and the slow-changing metadata poll loop.

## Manually running pieces

```bash
uv run stockwatch seed --n 20     # seed the watchlist
uv run stockwatch watchlist-count
uv run stockwatch poll --interval 3600   # metadata poll loop only
uv run stockwatch stream                 # producer + Flink job + stats consumer
uv run stockwatch run-once               # one metadata cycle, then detect + explain anomalies
```

## Tests

```bash
docker compose up -d postgres   # tests need a real Postgres, for the SCD2 partial-unique-index checks
uv run pytest
```

Everything is tested except the actual Gemini network call, which is stubbed out everywhere
(`llm/graph.py`'s `build_graph` takes the LLM client as a parameter for exactly this reason).

## Layout

See `src/stockwatch/`: `db/` (engine, models, the generic SCD2 upsert), `universe/` (ticker
watchlist, index constituents), `ingestion/` (yfinance pulls), `streaming/` (Redpanda/PyFlink
real-time price path), `features/`, `detection/`, `explain/` (SHAP), `llm/` (LangGraph), and
`pipeline/` (orchestration + CLI).
