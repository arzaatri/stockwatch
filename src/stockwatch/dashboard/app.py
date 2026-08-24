"""Streamlit dashboard: price history per selected ticker, with detected
anomalies marked (red exclamation marks). Clicking a marked point calls
explain_anomaly_at() for a point-in-time LLM explanation. Run via
`stockwatch dashboard` (which shells out to `streamlit run` on this file) -
not meaningfully unit-testable, so this is verified manually; all the logic
it depends on (detection/, pipeline/explain_anomaly.py) is unit-tested.
"""

from datetime import UTC, datetime, timedelta

import plotly.graph_objects as go
import polars as pl
import requests
import streamlit as st

from stockwatch.api import inference_client
from stockwatch.config import get_settings
from stockwatch.detection.model_store import is_model_stale
from stockwatch.detection.simple_detector import SimpleAnomalyDetector
from stockwatch.features.build_features import build_feature_matrix
from stockwatch.logging_utils import get_logger
from stockwatch.pipeline.explain_anomaly import explain_anomaly_at
from stockwatch.universe.watchlist import get_active_tickers

logger = get_logger(__name__)

ML_LABEL = "ML (IsolationForest)"
SIMPLE_LABEL = "Simple (3-sigma)"

LOOKBACK_OPTIONS: dict[str, timedelta | None] = {
    "1 day": timedelta(days=1),
    "7 days": timedelta(days=7),
    "30 days": timedelta(days=30),
    "90 days": timedelta(days=90),
    "All": None,
}


@st.cache_data(ttl=60)
def _load_scored_matrix(
    tickers: tuple[str, ...], lookback_label: str, detector_label: str
) -> tuple[pl.DataFrame, datetime | None]:
    """Returns (scored matrix, model trained_at). trained_at is only
    meaningful for the ML detector - None for Simple (no persisted model) or
    when the ML path fell back to an ad-hoc, unpersisted fit.
    """
    matrix = build_feature_matrix(list(tickers))
    lookback = LOOKBACK_OPTIONS[lookback_label]
    if lookback is not None:
        cutoff = datetime.now(UTC) - lookback
        matrix = matrix.filter(pl.col("window_end") >= cutoff)
    if matrix.is_empty():
        return matrix, None

    if detector_label == ML_LABEL:
        # ML scoring goes through the inference microservice, not an
        # in-process model load - a RequestException here covers both "not
        # enough rows for the service's ad-hoc fallback fit" (422) and "the
        # service is unreachable", either of which just means no ML score
        # for this selection yet.
        try:
            result = inference_client.score(matrix)
        except requests.RequestException:
            logger.exception("Inference service call failed while scoring for the dashboard")
            return matrix.clear(), None
        return result.scored_matrix, result.model_trained_at

    try:
        detector = SimpleAnomalyDetector()
        detector.fit(matrix)
    except ValueError:
        # e.g. not enough history yet for this ticker/lookback.
        logger.info(
            "Not enough rows (%d) to fit %s for this selection",
            matrix.height,
            detector_label,
        )
        return matrix.clear(), None
    return detector.score(matrix), None


@st.cache_data
def _cached_explanation(ticker: str, window_end: datetime) -> dict:
    return explain_anomaly_at(ticker, window_end)


def _build_chart(ticker: str, ticker_rows: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ticker_rows["window_end"].to_list(),
            y=ticker_rows["avg_price"].to_list(),
            mode="lines",
            name=ticker,
            hoverinfo="x+y",
        )
    )

    anomalies = ticker_rows.filter(pl.col("is_anomaly") == 1)
    if not anomalies.is_empty():
        fig.add_trace(
            go.Scatter(
                x=anomalies["window_end"].to_list(),
                y=anomalies["avg_price"].to_list(),
                mode="markers+text",
                marker={"size": 18, "color": "red", "opacity": 0.15},
                text=["❗"] * anomalies.height,
                textfont={"color": "red", "size": 18},
                textposition="middle center",
                name="Anomaly",
                # (ticker, window_end iso string) per point - only this trace
                # carries customdata, so a click resolves to an anomaly only
                # when the exclamation mark itself (not the price line) was clicked.
                customdata=[
                    [ticker, window_end.isoformat()]
                    for window_end in anomalies["window_end"].to_list()
                ],
                hovertemplate="Anomaly - click for explanation<extra></extra>",
            )
        )

    fig.update_layout(
        title=ticker, height=350, margin={"l": 20, "r": 20, "t": 40, "b": 20}
    )
    return fig


def _render_explanation(ticker: str, window_end: datetime) -> None:
    st.subheader(f"Explanation: {ticker} @ {window_end.isoformat()}")
    logger.info("Anomaly point clicked: %s @ %s", ticker, window_end)
    with st.spinner("Asking the LLM..."):
        try:
            result = _cached_explanation(ticker, window_end)
        except Exception as error:
            # surface any failure in the UI rather than crashing the app
            logger.exception("Explanation failed for %s @ %s", ticker, window_end)
            st.error(f"Explanation failed: {error}")
            return

    context = result["context"]
    explanation = result["explanation"]

    st.markdown(
        f"**{explanation.likely_cause_category}** (confidence: {explanation.confidence})"
    )
    st.write(explanation.summary)
    if explanation.supporting_evidence:
        st.write("Supporting evidence:")
        for item in explanation.supporting_evidence:
            st.write(f"- {item}")

    with st.expander("Point-in-time context used"):
        st.write(
            f"Sector: {context.sector or 'unknown'} / Industry: {context.industry or 'unknown'}"
        )
        if context.rating:
            rating = context.rating
            st.write(
                f"Rating as of this point: strong_buy={rating.strong_buy}, buy={rating.buy}, "
                f"hold={rating.hold}, sell={rating.sell}, strong_sell={rating.strong_sell}"
            )
        else:
            st.write("Rating: none available as of this point")
        if context.recent_news:
            st.write("News from the 24h before this point:")
            for item in context.recent_news:
                st.write(f"- [{item.scope}] {item.headline} ({item.publisher})")
        else:
            st.write("News: none available in the 24h before this point")


def main() -> None:
    st.set_page_config(page_title="stockwatch", layout="wide")
    st.title("stockwatch anomaly dashboard")

    tickers = get_active_tickers()
    with st.sidebar:
        selected = st.multiselect("Tickers", tickers, default=tickers[:3])
        lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS), index=1)
        detector_label = st.radio("Detector", [ML_LABEL, SIMPLE_LABEL])

    if not selected:
        st.info("Select at least one ticker.")
        return

    logger.info(
        "Loading dashboard for %s (lookback=%s, detector=%s)",
        selected,
        lookback_label,
        detector_label,
    )
    scored, trained_at = _load_scored_matrix(
        tuple(selected), lookback_label, detector_label
    )
    if scored.is_empty():
        st.warning("No price history yet for the selected tickers/lookback.")
        return

    if detector_label == ML_LABEL:
        if trained_at is None:
            st.info(
                "No trained model found - scoring with an ad-hoc fit for this "
                "view. Run `train_model.sh` to persist one."
            )
        else:
            max_age_days = get_settings().model_stale_after_days
            age_days = (datetime.now(UTC) - trained_at).days
            if is_model_stale(trained_at, max_age_days):
                st.warning(
                    f"Model was trained {age_days} day(s) ago (older than "
                    f"{max_age_days}) - consider running `train_model.sh` to refresh it."
                )
            else:
                st.caption(f"Model trained {trained_at.isoformat()} ({age_days}d ago)")

    for ticker in selected:
        ticker_rows = scored.filter(pl.col("ticker") == ticker).sort("window_end")
        if ticker_rows.is_empty():
            st.warning(f"No data for {ticker}.")
            continue

        fig = _build_chart(ticker, ticker_rows)
        event = st.plotly_chart(
            fig, on_select="rerun", key=f"chart_{ticker}", use_container_width=True
        )

        points = event["selection"]["points"] if event else []
        if points and points[0].get("customdata"):
            clicked_ticker, clicked_window_end_iso = points[0]["customdata"]
            _render_explanation(
                clicked_ticker, datetime.fromisoformat(clicked_window_end_iso)
            )


if __name__ == "__main__":
    main()
