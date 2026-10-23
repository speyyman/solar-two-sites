"""Everything the study fixes in advance: sites, window, inputs, seeds.

Nothing here is tuned on test data. Values come from the PVDAQ system metadata
(see DATA.md for the exact files) or are stated choices recorded in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

OEDI = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize"
PREVIOUS_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

SEED = 20241023
BOOTSTRAP_REPS = 1000
QUANTILES = (0.1, 0.5, 0.9)

# Hourly weather variables. Open-Meteo radiation values are means over the
# preceding hour, so every series in this study is labelled by hour end.
WEATHER_VARS = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
)

# Day-ahead inputs: the value each model predicted 24 h before the valid hour.
NWP_MODELS = {
    "gfs": "gfs_seamless",  # NOAA GFS with HRRR blended in over the US (primary)
    "ecmwf": "ecmwf_ifs025",  # ECMWF IFS 0.25 deg open data (robustness check)
}
PRIMARY_NWP = "gfs"


@dataclass(frozen=True)
class PvFile:
    """One measured file on OEDI and the columns summed to make the series."""

    key: str
    columns: tuple[str, ...]
    tz: str  # the clock the timestamps were written in (checked in clean.py)


@dataclass(frozen=True)
class Site:
    key: str
    system_id: int
    name: str
    place: str
    climate: str
    latitude: float
    longitude: float
    elevation_m: float
    dc_kw: float
    ac_kw: float
    tracking: bool
    tilt: float
    azimuth: float
    gamma_pdc: float  # power temperature coefficient, 1/C
    gcr: float
    tz: str
    start: str  # first local day used
    end: str  # last local day used
    pv_files: tuple[PvFile, ...]
    irradiance_files: tuple[PvFile, ...] = field(default=())
    notes: str = ""


SITES: dict[str, Site] = {
    "arbuckle": Site(
        key="arbuckle",
        system_id=2107,
        name="Farm Solar Array",
        place="Arbuckle, California",
        climate="Csa (hot-summer Mediterranean)",
        latitude=38.996306,
        longitude=-122.134111,
        elevation_m=10.0,
        # 2880 Hyundai HiS-M310TI modules (310 W) = 892.8 kW DC; metadata says 893.
        dc_kw=893.0,
        # 24 ABB TRIO-27.6-TL-OUTD inverters x 27.6 kW.
        ac_kw=24 * 27.6,
        tracking=False,
        tilt=25.0,
        azimuth=180.0,
        gamma_pdc=-0.0037,  # typical mono-Si; the metadata gives no coefficient
        gcr=0.4,
        start="2024-01-20",
        end="2025-12-31",
        tz="America/Los_Angeles",
        pv_files=(
            PvFile("2107_OEDI/data/2107_meter_15m_data_2024.csv",
                   ("meter_revenue_grade_ac_output_meter_149578",), "America/Los_Angeles"),
            PvFile("2107_OEDI/data/2107_meter_15m_data_2025.csv",
                   ("meter_revenue_grade_ac_output_meter_149578",), "America/Los_Angeles"),
        ),
        # The 2025 irradiance file is labelled UTC but follows Pacific standard time
        # (no daylight saving) until 2025-12; see clean.py and the paper's data section.
        irradiance_files=(
            PvFile("2107_OEDI/data/2107_irradiance_data_2024.csv",
                   ("poa_irradiance_o_149574",), "America/Los_Angeles"),
            PvFile("2107_OEDI/data/2107_irradiance_15m_data_2025.csv",
                   ("poa_irradiance_o_149574",), "Etc/GMT+8"),
        ),
        notes="Fixed ground mount, Sacramento Valley floor.",
    ),
    "kersey": Site(
        key="kersey",
        system_id=9068,
        name="SR_CO",
        place="Kersey, Colorado",
        climate="BSk (cold semi-arid)",
        latitude=40.3864,
        longitude=-104.5512,
        elevation_m=1407.0,
        # 40320 CdTe modules x 117.5 W = 4737.6 kW DC; metadata says 4738.
        dc_kw=4738.0,
        # Two inverters, 1910 kW AC each, per the metadata.
        ac_kw=2 * 1910.0,
        tracking=True,
        tilt=0.0,
        azimuth=180.0,
        gamma_pdc=-0.0028,  # from the module entry in the metadata (-0.28 %/C)
        gcr=0.4,
        start="2024-01-20",
        end="2025-04-29",
        tz="America/Denver",
        # The site's revenue meter is unusable in this period (values above 16 MW in
        # 2024-08/09, near zero 2024-10 to 2024-12), so the two inverters' AC power is summed.
        pv_files=(
            PvFile("9068_OEDI/data/9068_ac_power_data_20240101_20250430.csv",
                   ("inverter_1_ac_power_(kw)_inv_150143", "inverter_2_ac_power_(kw)_inv_150144"),
                   "America/Denver"),
        ),
        irradiance_files=(
            PvFile("9068_OEDI/data/9068_irradiance_data_20240101_20250430.csv",
                   ("pyranometer_(class_a)_pad_1_poa_irradiance_(w/m2)_o_149723",),
                   "America/Denver"),
        ),
        notes="Single-axis tracker, high plains east of the Front Range.",
    ),
}

METADATA_FILES = {
    "arbuckle": "2107_OEDI/metadata/2107_system_metadata.json",
    "kersey": "9068_OEDI/metadata/9068_system_metadata.json",
}

# Rolling origin: train on everything before the test month, test on that month.
FIRST_TEST_MONTH = "2024-07"
# Hours with the sun this far below zenith 90 or more are excluded from scoring.
MAX_ZENITH = 85.0
