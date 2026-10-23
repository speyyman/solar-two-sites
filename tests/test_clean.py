import numpy as np
import pandas as pd
import pvlib
import pytest

from pipeline.clean import Ledger, align_clock, combine, hourly, qc_power, stuck_mask, to_utc

LON, LAT = -122.13, 39.0


def clear_day_series(days: int, step_min: int, offset_min: int = 0) -> pd.Series:
    """Clear-sky GHI at the given step, with timestamps shifted by offset_min."""
    t = pd.date_range(
        "2024-06-01", periods=days * 24 * 60 // step_min, freq=f"{step_min}min", tz="UTC"
    )
    loc = pvlib.location.Location(LAT, LON)
    ghi = loc.get_clearsky(t)["ghi"]
    s = pd.Series(ghi.to_numpy(), index=t + pd.Timedelta(minutes=offset_min))
    return s


def test_to_utc_drops_ambiguous_fall_back_hour():
    t = pd.date_range("2024-11-03 00:00", "2024-11-03 03:00", freq="15min")
    s = pd.Series(1.0, index=t)
    led = Ledger("x")
    out = to_utc(s, "America/Los_Angeles", led, "power")
    assert str(out.index.tz) == "UTC"
    assert led.rows[0]["count"] == 4  # 01:00-01:45 happen twice that night
    assert out.index.is_unique


def test_align_clock_removes_whole_step_offset():
    s = clear_day_series(20, 15, offset_min=-30)  # labelled 30 min early
    led = Ledger("x")
    out = align_clock(s, LON, led, "power")
    assert (out.index[0] - s.index[0]) == pd.Timedelta(minutes=30)
    assert "shifted" in led.rows[0]["rule"]


def test_align_clock_leaves_correct_clock_alone():
    s = clear_day_series(20, 15)
    led = Ledger("x")
    out = align_clock(s, LON, led, "power")
    assert out.index.equals(s.index)
    assert led.rows == []


def test_qc_power_rules():
    t = pd.date_range("2024-06-01 12:00", periods=10, freq="5min", tz="UTC")
    s = pd.Series([-3.0, 10, 11, 900, 12, 13, 14, 15, 16, 17], index=t)
    led = Ledger("x")
    out = qc_power(s, ac_max_kw=100, ledger=led)
    assert out.iloc[0] == 0.0
    assert np.isnan(out.iloc[3])
    counts = {r["rule"][:8]: r["count"] for r in led.rows}
    assert counts["negative"] == 1 and counts["above 10"] == 1


def test_stuck_mask_flags_frozen_logger_not_zeros():
    t = pd.date_range("2024-06-01", periods=30, freq="5min", tz="UTC")
    v = np.r_[np.zeros(10), np.full(13, 42.0), np.arange(7.0)]
    m = stuck_mask(pd.Series(v, index=t), 60)
    assert m.sum() == 13
    assert not m.iloc[:10].any()


def test_hourly_requires_coverage_and_labels_hour_end():
    t = pd.date_range("2024-06-01 00:15", periods=8, freq="15min", tz="UTC")
    s = pd.Series([1.0, 2, 3, 4, 5, np.nan, np.nan, np.nan], index=t)
    led = Ledger("x")
    h = hourly(s, led, "power")
    assert h.loc["2024-06-01 01:00", "value"] == pytest.approx(2.5)
    assert np.isnan(h.loc["2024-06-01 02:00", "value"])  # 1 of 4 samples
    assert led.rows[0]["count"] == 1


def test_combine_keeps_first_and_counts_overlap():
    t = pd.date_range("2024-06-01", periods=4, freq="h", tz="UTC")
    a = pd.Series([1.0, 2, 3, 4], index=t)
    b = pd.Series([3.5, 4, 5], index=t[2:].append(pd.DatetimeIndex([t[-1] + pd.Timedelta("1h")])))
    led = Ledger("x")
    out = combine([a, b], led, "power")
    assert list(out) == [1, 2, 3, 4, 5]
    assert led.rows[0]["count"] == 2 and led.rows[0]["detail"] == "1 differ"
