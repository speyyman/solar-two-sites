"""Hourly feature tables for one site and one weather source.

The forecast is issued at the end of local day D-1 for every hour of day D. A
feature may use: weather values for day D that the source issued at least 24 h
ahead (Open-Meteo `previous_day1`), the same source's values for earlier hours,
sun geometry, and power measured up to the end of D-1. `assert_no_leakage`
checks the last rule on every table this module builds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clean import Ledger
from pipeline.config import INTERIM, MAX_ZENITH, Site
from pipeline.physical import ac_power, clear_sky_power, sun

SOURCES = ("gfs", "ecmwf", "era5")
CORE_INPUTS = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "wind_speed_10m",
)
FEATURES = (
    "p_phys_n",
    "p_cs_n",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "kt_f",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_ms",
    "precipitation",
    "frozen_72h",
    "zenith",
    "azimuth",
    "hour_sin",
    "hour_cos",
    "k_prev",
    "y_prev_n",
)
PERSIST_MAX_DAYS = 7
FROZEN_C = 1.0


def hourly_index(site: Site) -> pd.DatetimeIndex:
    start = pd.Timestamp(site.start, tz=site.tz)
    end = pd.Timestamp(site.end, tz=site.tz) + pd.Timedelta(days=1)
    return pd.date_range(
        start.tz_convert("UTC") + pd.Timedelta(hours=1), end.tz_convert("UTC"), freq="1h"
    )


def weather(site: Site, source: str) -> pd.DataFrame:
    name = "era5" if source == "era5" else f"nwp_{source}"
    return pd.read_parquet(INTERIM / f"{site.key}_{name}.parquet")


def previous_day(
    values: pd.Series, day: pd.Series, hour: pd.Series, ledger: Ledger, what: str
) -> pd.Series:
    """Value at the same local hour on the most recent earlier day that has one.

    Uses D-1 when it exists, otherwise D-2 and so on up to 7 days back. Every hour
    filled from a day older than D-1 is counted in the ledger.
    """
    table = pd.DataFrame({"v": values.to_numpy(), "day": day.to_numpy(), "hour": hour.to_numpy()})
    grid = table.pivot_table(index="day", columns="hour", values="v", aggfunc="mean")
    full_days = pd.date_range(grid.index.min(), grid.index.max(), freq="D")
    grid = grid.reindex(full_days)
    d1 = grid.shift(1)
    filled = grid.ffill(limit=PERSIST_MAX_DAYS).shift(1)
    n_filled = int((d1.isna() & filled.notna()).to_numpy().sum())
    ledger.add(
        "features",
        f"{what}: D-1 missing, filled from an older day (up to 7)",
        n_filled,
        "day-hours",
    )
    stacked = filled.stack(future_stack=True)
    look = pd.MultiIndex.from_arrays([table["day"], table["hour"]])
    return pd.Series(stacked.reindex(look).to_numpy(), index=values.index)


def daily_ratio(y: pd.Series, p_cs: pd.Series, day: pd.Series, daytime: pd.Series) -> pd.Series:
    """Measured energy over clear-sky energy for each complete day (NaN if any gap)."""
    t = pd.DataFrame({"y": y, "cs": p_cs, "day": day})[daytime]
    g = t.groupby("day")
    complete = g["y"].count() == g.size()
    ratio = g["y"].sum() / g["cs"].sum()
    return ratio.where(complete)


def build(site: Site, source: str, ledger: Ledger) -> pd.DataFrame:
    idx = hourly_index(site)
    s = sun(site, idx)
    wx = weather(site, source).reindex(idx)
    df = s.join(wx)
    df["wind_ms"] = df["wind_speed_10m"] / 3.6
    ok = df[list(CORE_INPUTS)].notna().all(axis=1)
    phys = ac_power(
        site,
        s,
        df["shortwave_radiation"].fillna(0),
        df["direct_normal_irradiance"].fillna(0),
        df["diffuse_radiation"].fillna(0),
        df["temperature_2m"].fillna(20),
        df["wind_ms"].fillna(1),
    )
    df["p_phys"] = phys.where(ok)
    df["inputs_ok"] = ok
    df["p_cs"] = clear_sky_power(site, s)
    power = pd.read_parquet(INTERIM / f"{site.key}_power.parquet")
    df["y"] = power["p_kw"].reindex(idx)

    local = idx.tz_convert(site.tz)
    df["day"] = local.tz_localize(None).normalize()
    df["hour"] = local.hour
    mid = idx - pd.Timedelta(minutes=30)
    solar_hour = (mid.hour + mid.minute / 60 + site.longitude / 15.0) % 24
    df["hour_sin"] = np.sin(2 * np.pi * solar_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * solar_hour / 24)
    df["daytime"] = df["zenith"] < MAX_ZENITH

    df["kt_f"] = (df["shortwave_radiation"] / df["ghi_cs"]).where(df["ghi_cs"] > 50).clip(0, 1.5)
    frozen = df["precipitation"].where(df["temperature_2m"] <= FROZEN_C, 0.0)
    df["frozen_72h"] = frozen.rolling(72, min_periods=1).sum()

    # Persistence inputs: only power measured on earlier days.
    ratio = daily_ratio(df["y"], df["p_cs"], df["day"], df["daytime"])
    full = pd.date_range(ratio.index.min(), ratio.index.max(), freq="D")
    r1 = ratio.reindex(full).shift(1)
    rf = ratio.reindex(full).ffill(limit=PERSIST_MAX_DAYS).shift(1)
    if source == "gfs":
        ledger.add(
            "features",
            "smart persistence: D-1 incomplete, ratio taken from an older day",
            int((r1.isna() & rf.notna()).sum()),
            "days",
        )
    df["k_prev"] = df["day"].map(rf)
    df["y_prev"] = previous_day(
        df["y"],
        df["day"],
        df["hour"],
        ledger if source == "gfs" else Ledger("discard"),
        "persistence",
    )
    df["p_phys_n"] = df["p_phys"] / site.ac_kw
    df["p_cs_n"] = df["p_cs"] / site.ac_kw
    df["y_prev_n"] = df["y_prev"] / site.ac_kw
    df["y_n"] = df["y"] / site.ac_kw
    assert_no_leakage(df)
    return df


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Persistence features for day D must equal measurements from strictly earlier days."""
    y = df.set_index(["day", "hour"])["y"]
    y = y[~y.index.duplicated()]
    sample = df.dropna(subset=["y_prev"]).iloc[:: max(1, len(df) // 500)]
    for _, row in sample.iterrows():
        # The value must exist on some earlier day at the same hour.
        for back in range(1, PERSIST_MAX_DAYS + 2):
            key = (row["day"] - pd.Timedelta(days=back), row["hour"])
            if key in y.index and not np.isnan(y.loc[key]):
                assert np.isclose(y.loc[key], row["y_prev"]), "persistence used the wrong day"
                break
        else:
            raise AssertionError("persistence value with no earlier measurement")
    days = df.groupby("day")["k_prev"].nunique(dropna=True)
    assert (days <= 1).all(), "smart persistence ratio varies within a day"
