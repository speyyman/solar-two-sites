"""A physical PV model built from the plant metadata alone (nothing fitted to data).

Chain: sun position at the middle of each hour, Ineichen clear sky, Hay-Davies
transposition onto the fixed plane or the single-axis tracker (with backtracking),
Fresnel incidence-angle loss on beam light, Faiman cell temperature, PVWatts DC with
the default PVWatts system losses, PVWatts inverter clipped at the inverters' maximum
AC output. Inputs are hour-mean irradiance labelled by hour end, as Open-Meteo serves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from pipeline.config import Site

ETA_INV = 0.96
MAX_TRACKER_ANGLE = 60.0


def mid_hour(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Hour-end labels to the middle of the hour they describe."""
    return index - pd.Timedelta(minutes=30)


def sun(site: Site, index: pd.DatetimeIndex) -> pd.DataFrame:
    loc = pvlib.location.Location(site.latitude, site.longitude, altitude=site.elevation_m)
    t = mid_hour(index)
    sp = loc.get_solarposition(t)
    cs = loc.get_clearsky(t, solar_position=sp)
    out = pd.DataFrame(
        {
            "zenith": sp["zenith"].to_numpy(),
            "apparent_zenith": sp["apparent_zenith"].to_numpy(),
            "azimuth": sp["azimuth"].to_numpy(),
            "ghi_cs": cs["ghi"].to_numpy(),
            "dni_cs": cs["dni"].to_numpy(),
            "dhi_cs": cs["dhi"].to_numpy(),
            "dni_extra": pvlib.irradiance.get_extra_radiation(t).to_numpy(),
        },
        index=index,
    )
    return out


def orientation(site: Site, s: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if not site.tracking:
        n = len(s)
        return (pd.Series(np.full(n, site.tilt), index=s.index),
                pd.Series(np.full(n, site.azimuth), index=s.index))
    tr = pvlib.tracking.singleaxis(
        s["apparent_zenith"], s["azimuth"], axis_tilt=0.0, axis_azimuth=180.0,
        max_angle=MAX_TRACKER_ANGLE, backtrack=True, gcr=site.gcr,
    )
    tilt = tr["surface_tilt"].fillna(0.0)
    az = tr["surface_azimuth"].fillna(180.0)
    return tilt, az


def ac_power(site: Site, s: pd.DataFrame, ghi: pd.Series, dni: pd.Series, dhi: pd.Series,
             temp_air: pd.Series, wind_ms: pd.Series) -> pd.Series:
    """AC power in kW for the given hour-mean irradiance and weather."""
    up = s["zenith"] < 90
    ghi = ghi.where(up, 0.0).clip(lower=0)
    dhi = dhi.where(up, 0.0).clip(lower=0)
    dni = dni.where(up, 0.0).clip(lower=0, upper=1100)
    tilt, az = orientation(site, s)
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, az, s["apparent_zenith"], s["azimuth"], dni, ghi, dhi,
        dni_extra=s["dni_extra"], model="haydavies",
    )
    aoi = pvlib.irradiance.aoi(tilt, az, s["apparent_zenith"], s["azimuth"])
    iam = pvlib.iam.physical(aoi).fillna(0.0)
    eff = (poa["poa_direct"].fillna(0) * iam + poa["poa_diffuse"].fillna(0)).clip(lower=0)
    t_cell = pvlib.temperature.faiman(poa["poa_global"].fillna(0), temp_air, wind_ms)
    loss = pvlib.pvsystem.pvwatts_losses() / 100.0
    pdc = pvlib.pvsystem.pvwatts_dc(eff, t_cell, site.dc_kw, site.gamma_pdc) * (1 - loss)
    pac = pvlib.inverter.pvwatts(pdc, site.ac_max_kw / ETA_INV, eta_inv_nom=ETA_INV)
    pac = pac.clip(lower=0, upper=site.ac_max_kw)
    return pac.where(up, 0.0).fillna(0.0)


def clear_sky_power(site: Site, s: pd.DataFrame) -> pd.Series:
    """AC power under clear sky at 20 C and 1 m/s wind: the denominator of smart persistence."""
    n = len(s)
    return ac_power(site, s, s["ghi_cs"], s["dni_cs"], s["dhi_cs"],
                    pd.Series(np.full(n, 20.0), index=s.index),
                    pd.Series(np.full(n, 1.0), index=s.index))
