# stockwatch

An explainable stock anomaly detection pipeline: real-time price streaming (Redpanda + PyFlink),
slow-changing metadata (sector, index membership, analyst ratings, splits, earnings, news) tracked
with CDC/SCD2 in Postgres, IsolationForest anomaly detection, SHAP explanations, and an LLM
(Gemini, via LangChain/LangGraph) that reasons about the likely cause.

Model serving is a separate microservice from everything else: `inference_service/` owns the
trained IsolationForest artifact and SHAP, is stateless/DB-agnostic, and is called over HTTP by
the orchestration `api/` service (Postgres, news/ratings, the LLM, async job tracking). They're
independently deployable/scalable - see `k8s/` for a minimal demo of that.

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
uv run stockwatch train-model            # fit + persist an IsolationForest
uv run stockwatch serve-inference        # isolation-forest microservice (default :8001)
uv run stockwatch serve                  # orchestration api (default :8000) - needs the above running
uv run stockwatch run-once               # one metadata cycle, then detect + explain anomalies
```

`start_app.sh` launches both `serve-inference` and `serve` for you. With them running:

```bash
curl http://localhost:8000/detect                                    # fast path, no LLM
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "window_end": "2026-01-05T12:00:00Z"}'       # -> {"job_id": ...}
curl http://localhost:8000/explain/<job_id>                          # poll for the LLM result
curl http://localhost:8000/models/current                            # model staleness/metadata
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
real-time price path), `features/`, `detection/` + `explain/` (SHAP) - the model logic itself,
served by `inference_service/` (the isolation-forest microservice: `/health`, `/model/current`,
`/score`) - `llm/` (LangGraph), `api/` (the orchestration service: Postgres, news/ratings, the
LLM, async `/explain` jobs; talks to `inference_service/` over HTTP via `api/inference_client.py`),
`dashboard/` (Streamlit), and `pipeline/` (batch orchestration + the `stockwatch` CLI).
