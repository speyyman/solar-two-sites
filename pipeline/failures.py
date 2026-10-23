"""Where the primary forecast failed, and why, by rules fixed before the cases were read.

For every scored day: the daily energy error of the primary model (GBM on GFS
day-ahead inputs), and three daily clearness indices on the same clear-sky base:
  k_obs   measured energy / clear-sky model energy (what the plant saw)
  k_nwp   day-ahead forecast GHI / clear-sky GHI (what the forecaster was told)
  k_era5  reanalysis GHI / clear-sky GHI (a later, independent estimate of the sky)
The worst days (largest absolute daily error) are sorted with the rules in
`categorise`. The rules use only these indices, ERA5 frozen precipitation and the
plant's power factor; the thresholds are round numbers, not tuned.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pipeline.clean import Ledger
from pipeline.config import INTERIM, PRIMARY_NWP, RAW, RESULTS, SITES, Site
from pipeline.evaluate import MAIN, POINT, REF, scored
from pipeline.features import build
from pipeline.metrics import bootstrap_skill, nrmse, skill

WORST_N = 15
GAP = 0.2  # clearness difference treated as a real disagreement
SUNNY = 0.6
NO_OUTPUT = 0.05
CLEAR = 0.9
DRY_RH = 35.0
STILL_MS = 1.5


KERSEY_INV = RAW / "kersey" / "9068_ac_power_data_20240101_20250430.csv"
INV1, INV2 = "inverter_1_ac_power_(kw)_inv_150143", "inverter_2_ac_power_(kw)_inv_150144"


def kersey_power_factor() -> pd.Series:
    """Daily median power factor of inverter 1 while it produces over 500 kW (local days)."""
    pf_col = "inverter_1_power_factor_inv_150145"
    df = pd.read_csv(KERSEY_INV, usecols=["measured_on", pf_col, INV1])
    t = pd.to_datetime(df["measured_on"])
    on = df[INV1] > 500
    pf = pd.Series(df.loc[on, pf_col].to_numpy(), index=t[on].dt.normalize())
    return pf.groupby(level=0).median()


def kersey_inverter_down() -> pd.Series:
    """True on local days when one inverter made under 10% of the other's energy."""
    df = pd.read_csv(KERSEY_INV, usecols=["measured_on", INV1, INV2])
    t = pd.to_datetime(df["measured_on"]).dt.normalize()
    e = df[[INV1, INV2]].clip(lower=0).groupby(t.to_numpy()).sum()
    hi, lo = e.max(axis=1), e.min(axis=1)
    return (lo < 0.1 * hi) & (hi > 0)


def plant_anomaly(d: pd.DataFrame) -> pd.Series:
    """Days the plant, not the sky, most likely limited output. Uses no forecast."""
    no_output = (d["k_obs"] < NO_OUTPUT) & (d["k_era5"] >= SUNNY)
    clear_dry = (
        (d["k_era5"] >= CLEAR)
        & (d["rh_era5_min"] < DRY_RH)
        & (d["k_obs"] < d["k_era5"] - 1.5 * GAP)
    )
    flag = no_output | clear_dry
    if "inverter_down" in d:
        flag |= d["inverter_down"].fillna(False).astype(bool)
    if "power_factor" in d:
        flag |= (
            (d["power_factor"] < 0.95)
            & (d["k_era5"] >= SUNNY)
            & (d["k_obs"] < d["k_era5"] - 2 * GAP)
        )
    return flag


def daily_table(site: Site, pred: pd.DataFrame) -> pd.DataFrame:
    s = scored(pred, POINT)
    led = Ledger("discard")
    nwp = build(site, PRIMARY_NWP, led).reindex(s.index)
    era = build(site, "era5", led).reindex(s.index)
    g = pd.DataFrame(
        {
            "day": s["day"],
            "y": s["y_n"],
            "f": s[MAIN],
            "sp": s["smart_persistence"],
            "cs": s["p_cs_n"],
            "ghi_cs": nwp["ghi_cs"],
            "ghi_nwp": nwp["shortwave_radiation"],
            "ghi_era5": era["shortwave_radiation"],
            "cloud_nwp": nwp["cloud_cover"],
            "cloud_era5": era["cloud_cover"],
            "rh_era5": era["relative_humidity_2m"],
            "wind_era5": era["wind_ms"],
            "frozen_era5": era["frozen_72h"],
        }
    ).groupby("day")
    d = pd.DataFrame(
        {
            "hours": g.size(),
            "energy_obs": g["y"].sum(),
            "energy_fc": g["f"].sum(),
            "k_obs": g["y"].sum() / g["cs"].sum(),
            "k_nwp": g["ghi_nwp"].sum() / g["ghi_cs"].sum(),
            "k_era5": g["ghi_era5"].sum() / g["ghi_cs"].sum(),
            "cloud_nwp": g["cloud_nwp"].mean(),
            "cloud_era5": g["cloud_era5"].mean(),
            "rh_era5_min": g["rh_era5"].min(),
            "wind_era5_mean": g["wind_era5"].mean(),
            "frozen_era5_72h": g["frozen_era5"].max(),
            "sq_err": g.apply(lambda x: float(((x["f"] - x["y"]) ** 2).sum())),
            "abs_err_hours": g.apply(lambda x: float((x["f"] - x["y"]).abs().sum())),
        }
    )
    d["err"] = d["energy_fc"] - d["energy_obs"]  # AC-rated hours; + means over-forecast
    d["site"] = site.key
    if site.key == "kersey":
        d["power_factor"] = kersey_power_factor().reindex(d.index)
        d["inverter_down"] = kersey_inverter_down().reindex(d.index)
    poa = pd.read_parquet(INTERIM / f"{site.key}_poa_daily.parquet")["poa_kwh_m2"]
    d["poa_kwh_m2"] = poa.reindex(d.index)
    return d


def categorise(r: pd.Series) -> str:
    """Rules, version 2. Version 1 (git history) had one 'plant' bin for every day the
    plant fell far below a sky ERA5 called sunny; reading the cases showed two different
    things in it, still winter days and dry clear summer days, so they are split out."""
    if r["err"] > 0:
        if r["k_obs"] < NO_OUTPUT and r["k_era5"] >= SUNNY:
            return "no output at all under a sunny sky"
        if bool(r.get("inverter_down", False)):
            return "one inverter offline"
        if (
            r.get("power_factor", 1.0) < 0.95
            and r["k_era5"] >= SUNNY
            and r["k_obs"] < r["k_era5"] - 2 * GAP
        ):
            return "reactive-power limit"
        low = r["k_era5"] >= SUNNY and r["k_obs"] < r["k_era5"] - 2 * GAP
        if r["frozen_era5_72h"] >= 1.0 and low:
            return "snow likely"
        if (
            r["k_nwp"] >= CLEAR
            and r["k_era5"] >= CLEAR
            and r["rh_era5_min"] < DRY_RH
            and r["k_obs"] < r["k_era5"] - 1.5 * GAP
        ):
            return "clear dry sky, low output"
        if (
            r["k_era5"] >= SUNNY
            and r["k_obs"] < r["k_era5"] - 2 * GAP
            and r["wind_era5_mean"] < STILL_MS
            and r.name.month in (11, 12, 1, 2)
        ):
            return "still winter day, reanalysis clear"
        if r["k_nwp"] - r["k_era5"] >= GAP:
            return "forecast missed cloud"
        if r["k_obs"] < r["k_era5"] - GAP:
            return "cloud neither source saw"
        return "over-forecast, no clear cause"
    if r["k_era5"] - r["k_nwp"] >= GAP:
        return "forecast missed sun"
    if r["k_obs"] > r["k_era5"] + GAP:
        return "sun neither source saw"
    return "under-forecast, no clear cause"


def run() -> None:
    out: dict = {"worst_n": WORST_N, "gap": GAP, "sunny": SUNNY, "sites": {}}
    frames = []
    for key, site in SITES.items():
        pred = pd.read_parquet(INTERIM / f"{key}_predictions.parquet")
        d = daily_table(site, pred)
        d["category"] = d.apply(categorise, axis=1)
        frames.append(d)
        worst = d.reindex(d["err"].abs().sort_values(ascending=False).index).head(WORST_N)
        total_abs = d["abs_err_hours"].sum()
        top10 = d["abs_err_hours"].sort_values(ascending=False)
        n10 = max(1, int(np.ceil(0.1 * len(d))))
        anomaly = plant_anomaly(d)
        s = scored(pred, POINT)
        keep = ~s["day"].map(anomaly).fillna(False).astype(bool)
        sk = s[keep]
        y, ref = sk["y_n"].to_numpy(), sk[REF].to_numpy()
        sens = {"days_removed": int(anomaly.sum()), "hours": int(keep.sum())}
        for c in (REF, f"physical_cal/{PRIMARY_NWP}", MAIN, "gbm/era5"):
            f = sk[c].to_numpy()
            sens[c] = {
                "nrmse": nrmse(y, f),
                "skill": skill(y, f, ref),
                "skill_ci95": bootstrap_skill(y, f, ref, sk["day"]) if c != REF else None,
            }
        out["sites"][key] = {
            "days": len(d),
            "without_plant_anomaly_days": sens,
            "plant_anomaly_days": [str(i.date()) for i in d.index[anomaly]],
            "share_abs_error_worst_10pct_days": float(top10.head(n10).sum() / total_abs),
            "share_sq_error_worst_10pct_days": float(
                d["sq_err"].sort_values(ascending=False).head(n10).sum() / d["sq_err"].sum()
            ),
            "worst_categories": worst["category"].value_counts().to_dict(),
            "all_days_categories_abs_err_gt_1h": d[d["err"].abs() > 1.0]["category"]
            .value_counts()
            .to_dict(),
            "days_abs_err_gt_1h": int((d["err"].abs() > 1.0).sum()),
            "worst": [
                {
                    "day": str(i.date()),
                    **{
                        k: (
                            None
                            if pd.isna(v)
                            else (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        )
                        for k, v in r.items()
                        if k != "site"
                    },
                }
                for i, r in worst.iterrows()
            ],
        }
        print(f"{key}: worst {WORST_N} days")
        cols = [
            "err",
            "energy_obs",
            "energy_fc",
            "k_obs",
            "k_nwp",
            "k_era5",
            "rh_era5_min",
            "frozen_era5_72h",
            "category",
        ]
        print(worst[cols].round(2).to_string())
        print(out["sites"][key]["all_days_categories_abs_err_gt_1h"])
    pd.concat(frames).to_csv(RESULTS / "daily_errors.csv")
    (RESULTS / "failures.json").write_text(json.dumps(out, indent=2) + "\n")
