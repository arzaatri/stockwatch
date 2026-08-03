"""Pure rolling-statistics math for the streaming price path - zero Flink/Kafka
imports on purpose, so this is plain-pytest-testable. flink_job.py's
RollingStatsProcessFunction is a thin adapter that stores/loads this state via
Flink's keyed ValueState and delegates the actual computation here.

Uses Welford's online algorithm so a ticker's running mean/variance can be
updated one window-average at a time without keeping the full history around.
"""

from pydantic import BaseModel

WINDOW_SECONDS = 60  # tumbling window size, shared by flink_job.py (live) and
# pipeline/backfill.py (batch replay), so both populate windowed_price_stats
# at the same granularity.


class TickerRunningStats(BaseModel):
    count: int = 0
    mean: float = 0.0
    m2: float = (
        0.0  # sum of squared differences from the running mean (Welford's algorithm)
    )

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def volatility(self) -> float:
        return self.variance**0.5


def update_running_stats(
    state: TickerRunningStats, new_value: float
) -> tuple[TickerRunningStats, float]:
    """Fold `new_value` into `state`, returning the updated state and the
    z-score of `new_value` against the *prior* state (i.e. "how unusual was
    this window's average, relative to this ticker's own recent history").
    """
    zscore = 0.0
    if state.count >= 2 and state.volatility > 0:
        zscore = (new_value - state.mean) / state.volatility

    count = state.count + 1
    delta = new_value - state.mean
    mean = state.mean + delta / count
    delta2 = new_value - mean
    m2 = state.m2 + delta * delta2

    return TickerRunningStats(count=count, mean=mean, m2=m2), zscore
