"""Rolling-origin evaluation.

For each test month M (local calendar), every learned forecaster is trained on all
days before M and predicts every hour of M. Persistence and the uncalibrated
physical model need no training. Every forecaster is scored on the same hours:
daytime (sun more than 5 degrees up), measured power present, and every forecast
present. Errors are fractions of the AC rating.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pipeline import models as m
from pipeline.clean import Ledger
from pipeline.config import FIRST_TEST_MONTH, INTERIM, PRIMARY_NWP, QUANTILES, RESULTS, SITES, Site
from pipeline.features import SOURCES, build
from pipeline.metrics import (
    bias,
    bootstrap_diff,
    bootstrap_skill,
    coverage,
    nmae,
    nrmse,
    pinball,
    skill,
)

COMMON_LAST_MONTH = "2025-04"  # last month both sites have data
SKY_BINS = ((0.0, 0.4, "overcast"), (0.4, 0.8, "mixed"), (0.8, 9.9, "clear"))
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
POINT = ("persistence", "smart_persistence",
         *(f"{k}/{s}" for s in SOURCES for k in ("physical", "physical_cal", "gbm")),
         f"gbm_transfer/{PRIMARY_NWP}")
REF = "smart_persistence"
MAIN = f"gbm/{PRIMARY_NWP}"


def test_months(site: Site) -> list[pd.Period]:
    first = pd.Period(FIRST_TEST_MONTH, "M")
    last = pd.Period(site.end[:7], "M")
    return list(pd.period_range(first, last, freq="M"))


def forecast_site(site: Site, tables: dict[str, pd.DataFrame],
                  other: pd.DataFrame, other_cap: float) -> pd.DataFrame:
    cap = site.ac_max_kw / site.ac_kw
    base = tables[PRIMARY_NWP]
    out = []
    for month in test_months(site):
        start = month.start_time
        in_month = base["day"].dt.to_period("M") == month
        rows = base[in_month]
        pred = pd.DataFrame(index=rows.index)
        pred["day"] = rows["day"]
        pred["y_n"] = rows["y_n"]
        pred["daytime"] = rows["daytime"]
        pred["p_cs_n"] = rows["p_cs_n"]
        pred["persistence"] = m.finish(m.persistence(rows), rows, cap)
        pred["smart_persistence"] = m.finish(m.smart_persistence(rows), rows, cap)
        for src, df in tables.items():
            train, test = df[df["day"] < start], df[in_month]
            ok = test["inputs_ok"].to_numpy()
            pred[f"physical/{src}"] = np.where(ok, m.finish(m.physical(test), test, cap), np.nan)
            pred[f"physical_cal/{src}"] = np.where(
                ok, m.finish(m.physical_calibrated(train, test), test, cap), np.nan)
            pred[f"gbm/{src}"] = np.where(ok, m.finish(m.gbm(train, test), test, cap), np.nan)
            q = m.finish(m.gbm_quantiles(train, test), test, cap)
            for j, a in enumerate(QUANTILES):
                pred[f"gbm_q{int(a * 100)}/{src}"] = np.where(ok, q[:, j], np.nan)
            pred[f"wx_kt/{src}"] = test["kt_f"]
            pred[f"wx_cloud/{src}"] = test["cloud_cover"]
        # Transfer: trained only on the other site, same cut-off date.
        tr = other[other["day"] < start]
        test = tables[PRIMARY_NWP][in_month]
        ok = test["inputs_ok"].to_numpy()
        if len(m.trainable(tr)) > 500:
            pred[f"gbm_transfer/{PRIMARY_NWP}"] = np.where(
                ok, m.finish(m.gbm(tr, test), test, cap), np.nan)
        else:
            pred[f"gbm_transfer/{PRIMARY_NWP}"] = np.nan
        pred["month"] = str(month)
        out.append(pred)
    del other_cap
    return pd.concat(out)


def scored(pred: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    mask = pred["daytime"] & pred["y_n"].notna()
    for c in cols:
        mask &= pred[c].notna()
    return pred[mask]


def sky_class(pred: pd.DataFrame) -> pd.Series:
    """Daily measured energy over clear-sky energy, binned."""
    g = pred.groupby("day")
    k = g["y_n"].sum() / g["p_cs_n"].sum()
    labels = pd.Series(index=k.index, dtype=object)
    for lo, hi, name in SKY_BINS:
        labels[(k >= lo) & (k < hi)] = name
    return pred["day"].map(labels)


def point_table(s: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    y, ref = s["y_n"].to_numpy(), s[REF].to_numpy()
    res = {}
    for c in cols:
        f = s[c].to_numpy()
        res[c] = {"nmae": nmae(y, f), "nrmse": nrmse(y, f), "bias": bias(y, f),
                  "skill": skill(y, f, ref)}
    return res


def quantile_table(s: pd.DataFrame) -> dict[str, dict[str, float]]:
    res = {}
    y = s["y_n"].to_numpy()
    for src in SOURCES:
        q = s[[f"gbm_q{int(a * 100)}/{src}" for a in QUANTILES]].to_numpy()
        res[src] = {"pinball": pinball(y, q), "coverage_80": coverage(y, q[:, 0], q[:, 2]),
                    "width_80": float(np.mean(q[:, 2] - q[:, 0]))}
    return res


def summarise(pred: pd.DataFrame, site: Site, months: tuple[str, str] | None) -> dict:
    p = pred
    if months:
        p = pred[(pred["month"] >= months[0]) & (pred["month"] <= months[1])]
    qcols = tuple(f"gbm_q{int(a * 100)}/{s}" for s in SOURCES for a in QUANTILES)
    s = scored(p, (*POINT, *qcols))
    y, ref = s["y_n"].to_numpy(), s[REF].to_numpy()
    out: dict = {
        "site": site.key, "hours": len(s), "days": int(s["day"].nunique()),
        "first_day": str(s["day"].min().date()), "last_day": str(s["day"].max().date()),
        "mean_daytime_power_n": float(y.mean()),
        "point": point_table(s, POINT),
        "quantile": quantile_table(s),
    }
    ci = {}
    for c in POINT:
        if c == REF:
            continue
        ci[c] = bootstrap_skill(y, s[c].to_numpy(), ref, s["day"])
    out["skill_ci95"] = ci
    # Does learning beat the physical model it starts from? Interval for the RMSE gap.
    out["gbm_minus_physical_cal_rmse_ci95"] = {
        src: bootstrap_diff(y, s[f"gbm/{src}"].to_numpy(), s[f"physical_cal/{src}"].to_numpy(),
                            s["day"]) for src in SOURCES}
    out["gbm_minus_transfer_rmse_ci95"] = bootstrap_diff(
        y, s[MAIN].to_numpy(), s[f"gbm_transfer/{PRIMARY_NWP}"].to_numpy(), s["day"])
    sky = sky_class(s)
    out["by_sky"] = {}
    for _, _, name in SKY_BINS:
        part = s[sky == name]
        if len(part):
            out["by_sky"][name] = {"hours": len(part), "days": int(part["day"].nunique()),
                                   "point": point_table(part, POINT)}
    season = s["day"].dt.month.map(SEASONS)
    out["by_season"] = {}
    for name in ("DJF", "MAM", "JJA", "SON"):
        part = s[season == name]
        if len(part):
            out["by_season"][name] = {"hours": len(part), "days": int(part["day"].nunique()),
                                      "point": point_table(part, POINT)}
    out["by_month"] = {
        mo: {"hours": len(g), "point": point_table(g, (REF, f"physical_cal/{PRIMARY_NWP}", MAIN))}
        for mo, g in s.groupby("month")
    }
    return out


def run_all() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    ledgers = []
    for key, site in SITES.items():
        led = Ledger(key)
        tables[key] = {src: build(site, src, led) for src in SOURCES}
        ledgers.append(led.frame())
    pd.concat(ledgers).to_csv(RESULTS / "ledger_features.csv", index=False)
    keys = list(SITES)
    summary: dict = {"primary_nwp": PRIMARY_NWP, "reference": REF, "sites": {}}
    for i, key in enumerate(keys):
        site = SITES[key]
        other_key = keys[1 - i]
        pred = forecast_site(site, tables[key], tables[other_key][PRIMARY_NWP],
                             SITES[other_key].ac_kw)
        pred.to_parquet(INTERIM / f"{key}_predictions.parquet")
        summary["sites"][key] = {
            "full": summarise(pred, site, None),
            "common": summarise(pred, site, (FIRST_TEST_MONTH, COMMON_LAST_MONTH)),
        }
        f = summary["sites"][key]["full"]
        print(f"{key}: {f['hours']} hours, {f['days']} days")
        for c, v in f["point"].items():
            lo, hi = f["skill_ci95"].get(c, (0.0, 0.0))
            print(f"  {c:28s} nMAE {v['nmae']:.4f} nRMSE {v['nrmse']:.4f} "
                  f"bias {v['bias']:+.4f} skill {v['skill']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for src, q in f["quantile"].items():
            print(f"  quantiles {src}: pinball {q['pinball']:.4f} "
                  f"coverage {q['coverage_80']:.3f} width {q['width_80']:.3f}")
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
