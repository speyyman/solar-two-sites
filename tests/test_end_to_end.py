"""End to end on a synthetic plant: features -> models -> rolling-origin scores.

The synthetic truth is the physical model on a 'true' sky times 0.9 plus noise;
the 'forecast' sky is the true sky with error. A correct pipeline must then rank
hindsight weather above the forecast and both above smart persistence, keep night
at zero, and never let a feature see the day it predicts.
"""

import dataclasses

import numpy as np
import pandas as pd
import pvlib
import pytest

from pipeline import evaluate, features
from pipeline.clean import Ledger
from pipeline.config import SITES
from pipeline.features import assert_no_leakage, build, hourly_index
from pipeline.metrics import nrmse
from pipeline.physical import ac_power, sun

SITE = dataclasses.replace(SITES["arbuckle"], key="synthetic", start="2024-01-20", end="2024-08-31")


def weather(idx: pd.DatetimeIndex, kt_day: pd.Series) -> pd.DataFrame:
    s = sun(SITE, idx)
    kt = idx.tz_convert(SITE.tz).tz_localize(None).normalize().map(kt_day).to_numpy()
    ghi = s["ghi_cs"].to_numpy() * kt
    doy = (idx - pd.Timedelta(minutes=30)).day_of_year.to_numpy()
    parts = pvlib.irradiance.erbs(ghi, s["zenith"].to_numpy(), doy)
    return pd.DataFrame(
        {
            "shortwave_radiation": ghi,
            "direct_normal_irradiance": parts["dni"],
            "diffuse_radiation": parts["dhi"],
            "cloud_cover": 100 * (1 - kt),
            "temperature_2m": 20.0,
            "relative_humidity_2m": 50.0,
            "wind_speed_10m": 10.0,
            "precipitation": 0.0,
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("interim")
    mp = pytest.MonkeyPatch()
    mp.setattr(features, "INTERIM", tmp)
    rng = np.random.default_rng(7)
    idx = hourly_index(SITE)
    days = pd.date_range(SITE.start, SITE.end, freq="D")
    kt_true = pd.Series(np.clip(rng.beta(4, 1.5, len(days)), 0.05, 1.0), index=days)
    kt_fc = (kt_true + rng.normal(0, 0.15, len(days))).clip(0.05, 1.0)
    truth = weather(idx, kt_true)
    fc = weather(idx, kt_fc)
    s = sun(SITE, idx)
    p = ac_power(
        SITE,
        s,
        truth["shortwave_radiation"],
        truth["direct_normal_irradiance"],
        truth["diffuse_radiation"],
        truth["temperature_2m"],
        truth["wind_speed_10m"] / 3.6,
    )
    y = (0.9 * p + rng.normal(0, 5, len(p))).clip(lower=0).where(s["zenith"] < 90, 0.0)
    pd.DataFrame({"p_kw": y, "coverage": 1.0}).to_parquet(tmp / "synthetic_power.parquet")
    fc.to_parquet(tmp / "synthetic_nwp_gfs.parquet")
    fc.to_parquet(tmp / "synthetic_nwp_ecmwf.parquet")
    truth.to_parquet(tmp / "synthetic_era5.parquet")
    led = Ledger("synthetic")
    t = {src: build(SITE, src, led) for src in features.SOURCES}
    yield t, led
    mp.undo()


def test_scores_rank_as_the_construction_demands(tables):
    t, _ = tables
    pred = evaluate.forecast_site(SITE, t, t["gfs"], SITE.ac_kw)
    s = evaluate.scored(pred, evaluate.POINT)
    assert len(s) > 600  # two test months of daylight hours
    y = s["y_n"].to_numpy()
    err = {c: nrmse(y, s[c].to_numpy()) for c in evaluate.POINT}
    assert err["gbm/era5"] < err["gbm/gfs"] < err["smart_persistence"]
    assert err["physical_cal/era5"] < err["physical/era5"]  # calibration finds the 0.9
    assert (pred.loc[~pred["daytime"], "gbm/gfs"] == 0).all()
    q = pred[[f"gbm_q{a}/gfs" for a in (10, 50, 90)]].dropna().to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()


def test_tables_pass_leakage_check_and_fail_when_leaked(tables):
    t, _ = tables
    df = t["gfs"].copy()
    assert_no_leakage(df)
    df["y_prev"] = df["y"]  # same-day value: a leak
    with pytest.raises(AssertionError):
        assert_no_leakage(df)


def test_persistence_comes_from_the_previous_day(tables):
    t, _ = tables
    df = t["gfs"]
    row = df[(df["day"] == "2024-05-10") & (df["hour"] == 12)].iloc[0]
    prev = df[(df["day"] == "2024-05-09") & (df["hour"] == 12)].iloc[0]
    assert row["y_prev"] == pytest.approx(prev["y"])
