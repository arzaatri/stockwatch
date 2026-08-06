from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.orm import Session

from stockwatch.db.models import Watchlist, WindowedPriceStats
from stockwatch.detection import model_store
from stockwatch.pipeline.train_model import train_and_save_model


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path / "models")


def _seed_price_history(db_session: Session, ticker: str, n: int) -> None:
    now = datetime.now(UTC)
    rng = np.random.default_rng(0)
    db_session.add(Watchlist(ticker=ticker, added_at=now, is_active=True))
    db_session.add_all(
        WindowedPriceStats(
            ticker=ticker,
            window_end=now - timedelta(minutes=n - i),
            avg_price=100.0 + float(rng.normal()),
            total_volume=1000,
            volatility_estimate=0.1,
            price_zscore=float(rng.normal()),
            ingested_at=now,
        )
        for i in range(n)
    )
    db_session.commit()


def test_train_and_save_model_persists_a_usable_model(db_session: Session) -> None:
    _seed_price_history(db_session, "AAPL", n=15)

    path = train_and_save_model()

    assert path.exists()
    loaded = model_store.load_latest_detector()
    assert loaded is not None
    detector, _trained_at = loaded
    assert detector.model is not None


def test_train_and_save_model_raises_below_min_rows(db_session: Session) -> None:
    _seed_price_history(db_session, "AAPL", n=3)

    with pytest.raises(ValueError, match="at least"):
        train_and_save_model()
