"""Clean the measured series and record every value changed or dropped.

Rules, in order, for each power file:
1. Parse the naive timestamps in the clock the file was written in (Site/PvFile.tz),
   drop the rows that clock makes ambiguous (the repeated night hour each November),
   convert to UTC.
2. Check the clock against the sun: the midpoint between the first and last sample
   above 5% of the daily peak should sit at solar noon. Any whole-step residual is
   removed by shifting the series, and the shift is written to the ledger.
3. Negative power (night-time tare) becomes 0. Power above 105% of the inverters'
   maximum AC output is physically impossible and is dropped. Runs of an identical
   non-zero value lasting an hour or more (a frozen logger) are dropped.
4. Average to hours labelled by hour end, to match the weather data. An hour needs
   75% of its samples, otherwise it is missing. Target values are never filled in.

Weather files are reshaped only (renamed columns, UTC index); nothing is filled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from pipeline.config import INTERIM, RAW, RESULTS, SITES, WEATHER_VARS, PvFile, Site

MIN_COVERAGE = 0.75
STUCK_MINUTES = 60
OVER_CAPACITY = 1.05
CLOCK_TOLERANCE_H = 0.75


@dataclass
class Ledger:
    """Every value the pipeline drops, changes or fills, with the rule that did it."""

    site: str
    rows: list[dict[str, object]] = field(default_factory=list)

    def add(self, stage: str, rule: str, count: int, unit: str, detail: str = "") -> None:
        self.rows.append(
            {
                "site": self.site,
                "stage": stage,
                "rule": rule,
                "count": int(count),
                "unit": unit,
                "detail": detail,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["site", "stage", "rule", "count", "unit", "detail"])


def read_file(path: Path, f: PvFile) -> pd.Series:
    """Sum the named columns; a timestamp missing any column is missing."""
    time_col = pd.read_csv(path, nrows=0).columns[0]
    df = pd.read_csv(path, usecols=[time_col, *f.columns])
    t = pd.to_datetime(df[time_col])
    values = df[list(f.columns)].sum(axis=1, min_count=len(f.columns))
    return pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(t), name=path.stem)


def to_utc(s: pd.Series, tz: str, ledger: Ledger, stage: str) -> pd.Series:
    local = s.index.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    bad = local.isna()
    if bad.any():
        ledger.add(
            stage,
            "timestamp ambiguous or nonexistent in local clock; dropped",
            int(bad.sum()),
            "samples",
            tz,
        )
    out = pd.Series(s.to_numpy()[~bad], index=local[~bad].tz_convert("UTC"), name=s.name)
    return out[~out.index.duplicated(keep="first")]


def solar_noon_utc_hours(days: pd.DatetimeIndex, longitude: float) -> np.ndarray:
    eot = pvlib.solarposition.equation_of_time_spencer71(days.day_of_year)
    return 12.0 - longitude / 15.0 - np.asarray(eot) / 60.0


def clock_residuals(s: pd.Series, longitude: float, frac: float = 0.05) -> pd.Series:
    """Per UTC-day offset (hours) between the production midpoint and solar noon.

    For the sites here (longitude -104 to -122) the whole daylight period falls
    inside one UTC day only for part of the year, so days are grouped by the
    local solar day instead: UTC time shifted by the longitude.
    """
    s = s.dropna()
    shift = pd.Timedelta(hours=longitude / 15.0)  # local mean solar time = UTC + lon/15
    solar = s.copy()
    solar.index = s.index + shift  # approximate local solar time, still tz-aware
    # Thresholds are relative to the record's high level (99.5th percentile), and only
    # days that reach 30% of it count, so night-time noise on dull days cannot set them.
    level = float(solar.quantile(0.995))
    peak = solar.groupby(solar.index.normalize()).transform("max")
    # A day counts only if it starts and ends in darkness, so a record that begins or
    # ends mid-day (or a gap) cannot fake a midpoint.
    day = solar.groupby(solar.index.normalize())
    dark_ends = (day.transform("first") < frac * level) & (day.transform("last") < frac * level)
    on = solar[(solar > frac * level) & (peak > 0.3 * level) & dark_ends]
    g = on.groupby(on.index.normalize())
    first, last, n = g.apply(lambda x: x.index.min()), g.apply(lambda x: x.index.max()), g.size()
    mid = first + (last - first) / 2
    hours = (mid - mid.index).dt.total_seconds() / 3600.0
    days = pd.DatetimeIndex(mid.index)
    # Solar noon in this shifted clock is 12 h minus the equation of time.
    noon_solar = solar_noon_utc_hours(days, longitude) + longitude / 15.0
    r = pd.Series(hours.to_numpy() - noon_solar, index=days.tz_localize(None))
    return r[n.reindex(days).to_numpy() >= 8]


def median_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    return pd.Series(index).diff().median()


def align_clock(s: pd.Series, longitude: float, ledger: Ledger, stage: str) -> pd.Series:
    """Remove a constant whole-step clock offset; drop months that disagree."""
    step = median_step(s.index)
    r = clock_residuals(s, longitude)
    resid_h = float(r.median())
    steps = int(round(resid_h * 3600 / step.total_seconds()))
    if steps:
        s = s.copy()
        s.index = s.index - steps * step
        ledger.add(
            stage,
            "clock shifted to match solar noon",
            int(s.notna().sum()),
            "samples",
            f"median residual {resid_h:+.2f} h; shifted by {-steps * step}",
        )
    monthly = (r - steps * step.total_seconds() / 3600).groupby(r.index.to_period("M")).median()
    bad_months = monthly[monthly.abs() > CLOCK_TOLERANCE_H]
    for m in bad_months.index:
        in_month = (s.index + pd.Timedelta(hours=longitude / 15.0)).tz_convert(None).to_period(
            "M"
        ) == m
        ledger.add(
            stage,
            "month whose clock disagrees with the sun by more than 0.75 h; dropped",
            int(np.count_nonzero(in_month)),
            "samples",
            f"{m}: residual {bad_months[m]:+.2f} h",
        )
        s = s[~in_month]
    return s


def stuck_mask(s: pd.Series, minutes: int) -> pd.Series:
    """True where an identical non-zero value repeats for `minutes` or longer."""
    step = median_step(s.index)
    need = max(2, int(round(pd.Timedelta(minutes=minutes) / step)))
    v = s.to_numpy()
    change = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(change)
    sizes = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    return pd.Series((sizes >= need) & (v > 0) & ~np.isnan(v), index=s.index)


def qc_power(s: pd.Series, ac_max_kw: float, ledger: Ledger) -> pd.Series:
    s = s.copy()
    neg = s < 0
    ledger.add("power", "negative power (night tare) set to 0", int(neg.sum()), "samples")
    s[neg] = 0.0
    over = s > OVER_CAPACITY * ac_max_kw
    ledger.add(
        "power",
        "above 105% of maximum AC output; dropped",
        int(over.sum()),
        "samples",
        f"limit {OVER_CAPACITY * ac_max_kw:.0f} kW",
    )
    s[over] = np.nan
    stuck = stuck_mask(s, STUCK_MINUTES)
    ledger.add(
        "power", "identical non-zero value for 60 min or more; dropped", int(stuck.sum()), "samples"
    )
    s[stuck] = np.nan
    return s


def hourly(s: pd.Series, ledger: Ledger, stage: str) -> pd.DataFrame:
    step = median_step(s.index)
    expected = pd.Timedelta(hours=1) / step
    g = s.resample("1h", label="right", closed="right")
    out = pd.DataFrame({"value": g.mean(), "coverage": g.count() / expected})
    thin = (out.coverage < MIN_COVERAGE) & out.value.notna()
    ledger.add(stage, "hour with under 75% of its samples; set missing", int(thin.sum()), "hours")
    out.loc[out.coverage < MIN_COVERAGE, "value"] = np.nan
    return out


def combine(parts: list[pd.Series], ledger: Ledger, stage: str) -> pd.Series:
    """Join files; where two files overlap, keep the first and log disagreements."""
    s = parts[0]
    for nxt in parts[1:]:
        both = s.index.intersection(nxt.index)
        if len(both):
            a, b = s.reindex(both), nxt.reindex(both)
            differ = ~np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
            ledger.add(
                stage,
                "overlapping timestamps across files; first file kept",
                len(both),
                "samples",
                f"{int(differ.sum())} differ",
            )
        s = pd.concat([s, nxt[~nxt.index.isin(s.index)]]).sort_index()
    return s


def clean_power(site: Site, ledger: Ledger) -> pd.DataFrame:
    parts = []
    for f in site.pv_files:
        raw = read_file(RAW / site.key / Path(f.key).name, f)
        ledger.add("power", "raw samples read", int(raw.notna().sum()), "samples", Path(f.key).name)
        missing = raw.isna()
        if missing.any():
            ledger.add(
                "power", "sample missing in source", int(missing.sum()), "samples", Path(f.key).name
            )
        parts.append(to_utc(raw, f.tz, ledger, "power"))
    s = combine(parts, ledger, "power")
    s = align_clock(s, site.longitude, ledger, "power")
    s = qc_power(s, site.ac_max_kw, ledger)
    h = hourly(s, ledger, "power").rename(columns={"value": "p_kw"})
    return h


def clean_poa(site: Site, ledger: Ledger) -> pd.Series:
    """Daily plane-of-array insolation (kWh/m2) per local day; diagnostic only."""
    parts = []
    for f in site.irradiance_files:
        raw = read_file(RAW / site.key / Path(f.key).name, f)
        s = to_utc(raw, f.tz, ledger, "poa")
        s = align_clock(s, site.longitude, ledger, "poa")
        parts.append(s)
    s = combine(parts, ledger, "poa").clip(lower=0)
    h = hourly(s, ledger, "poa")["value"]
    local = h.tz_convert(site.tz)
    day = local.groupby(local.index.normalize().tz_localize(None))
    daily = day.sum(min_count=20) / 1000.0
    return daily.rename("poa_kwh_m2")


def read_weather(path: Path, suffix: str) -> pd.DataFrame:
    j = json.loads(path.read_text())
    df = pd.DataFrame(j["hourly"])
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("time")), tz="UTC")
    df.index = idx
    df.columns = [c.removesuffix(suffix) for c in df.columns]
    return df[list(WEATHER_VARS)].astype(float)


def clean_site(site: Site) -> pd.DataFrame:
    ledger = Ledger(site.key)
    power = clean_power(site, ledger)
    power.to_parquet(INTERIM / f"{site.key}_power.parquet")
    poa = clean_poa(site, ledger)
    poa.to_frame().to_parquet(INTERIM / f"{site.key}_poa_daily.parquet")
    d = RAW / site.key
    for src in ("nwp_gfs", "nwp_ecmwf"):
        read_weather(d / f"{src}.json", "_previous_day1").to_parquet(
            INTERIM / f"{site.key}_{src}.parquet"
        )
    read_weather(d / "era5.json", "").to_parquet(INTERIM / f"{site.key}_era5.parquet")
    return ledger.frame()


def clean_all() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frames = [clean_site(s) for s in SITES.values()]
    led = pd.concat(frames, ignore_index=True)
    led.to_csv(RESULTS / "ledger_clean.csv", index=False, lineterminator="\n")
    print(led.to_string())
