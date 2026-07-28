"""Pure math, no Flink/Kafka/Postgres involved - these run without docker."""

import math

from stockwatch.streaming.rolling_stats import TickerRunningStats, update_running_stats


def test_first_value_has_zero_zscore() -> None:
    state, zscore = update_running_stats(TickerRunningStats(), 100.0)

    assert zscore == 0.0
    assert state.count == 1
    assert state.mean == 100.0


def test_running_mean_matches_hand_computed_average() -> None:
    values = [10.0, 12.0, 11.0, 13.0]
    state = TickerRunningStats()
    for value in values:
        state, _ = update_running_stats(state, value)

    assert math.isclose(state.mean, sum(values) / len(values))


def test_zscore_flags_a_clear_outlier() -> None:
    state = TickerRunningStats()
    for value in [10.0, 10.1, 9.9, 10.05, 9.95, 10.0, 10.02]:
        state, _ = update_running_stats(state, value)

    _, zscore = update_running_stats(state, 100.0)

    assert abs(zscore) > 5


def test_variance_and_volatility_are_nonnegative() -> None:
    state = TickerRunningStats()
    for value in [1.0, 5.0, 2.0, 8.0]:
        state, _ = update_running_stats(state, value)

    assert state.variance >= 0
    assert state.volatility >= 0
    assert math.isclose(state.volatility, state.variance**0.5)
