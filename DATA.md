# Data

Every input is public and downloaded by `python -m pipeline fetch`. Raw files go to `data/raw/` and are not committed; `data/manifest.json` (committed) holds the exact URL of every request and a SHA-256 of every file, and the fetch step stops if a file does not match (set `UPDATE_MANIFEST=1` to accept changed data on purpose; `git diff data/manifest.json` then shows what changed). All sources were accessed on 2026-10-23.

The repository redistributes no raw data. It does contain derived results (scores, daily errors, the cleaning ledger, figures), which the licences below allow with attribution.

## Measured plant output

**NREL Photovoltaic Data Acquisition (PVDAQ) Public Datasets**, Open Energy Data Initiative.
Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., Jordan, D. (2021). doi:[10.25984/1846021](https://doi.org/10.25984/1846021). Landing page: <https://data.openei.org/submissions/4568>.
Licence: Creative Commons Attribution 4.0 (stated in the dataset's metadata on the landing page).
Files read from the public bucket `https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/`, no key needed:

| Site | System | Files used |
|---|---|---|
| Arbuckle, California | 2107 | `2107_OEDI/data/2107_meter_15m_data_2024.csv`, `..._2025.csv` (revenue meter, 15 min); `2107_irradiance_data_2024.csv`, `2107_irradiance_15m_data_2025.csv` (on-site POA, checked, not used for results); `2107_OEDI/metadata/2107_system_metadata.json` |
| Kersey, Colorado | 9068 | `9068_OEDI/data/9068_ac_power_data_20240101_20250430.csv` (per-inverter AC power, power factor, revenue meter); `9068_irradiance_data_20240101_20250430.csv` (checked, not used); `9068_OEDI/metadata/9068_system_metadata.json` |

Why these two: the study needs measured output in 2024 and 2025, the period for which true day-ahead weather forecasts are archived (see below). Of the PVDAQ systems surveyed in October 2026 (the 2023 Solar Data Prize systems and the long-running public systems), these two had usable plant-level output in that period with documented metadata. The survey was not exhaustive, and other systems may qualify.

Known problems, all handled by stated rules in `pipeline/clean.py` and counted in `results/ledger_clean.csv`:

- Arbuckle's meter labels each 15-minute interval by its start; the clock check against solar noon found this and the series is shifted by 15 minutes.
- Arbuckle's 2025 irradiance file is labelled UTC but follows a fixed Pacific standard-time clock for most of the year; December 2024 and December 2025 disagree with the sun and are dropped. Irradiance is not used for any result.
- Kersey's revenue meter reports impossible values (several times the plant rating) in August and September 2024 and near zero from October to December 2024, while both inverters report normal output. Kersey's output is the sum of its two inverters' AC power.
- Kersey's irradiance sensor implies a performance ratio above one in August 2024; July 2024 fails the clock check and is dropped. Not used for any result.

## Inverter rating (Arbuckle)

ABB, *TRIO-20.0/27.6-TL-OUTD string inverter* datasheet, document 9AKK106103A4856: <https://search-ext.abb.com/library/Download.aspx?DocumentID=9AKK106103A4856>. Used for two numbers only: 27,600 W nominal and 30,000 W maximum AC output per inverter (24 inverters, per the PVDAQ metadata). Checked 2026-10-23.

## Weather

All weather comes through **Open-Meteo** (Zippenfenig, P., *Open-Meteo.com Weather API*, Zenodo, doi:[10.5281/zenodo.7970649](https://doi.org/10.5281/zenodo.7970649)). Licence of the API data: Creative Commons Attribution 4.0 (<https://open-meteo.com/en/licence>). The free API is offered for non-commercial use; this is public, non-commercial research. Hourly variables: `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `cloud_cover`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation`, all at the plant's coordinates (from the PVDAQ metadata), UTC, 2024-01-01 to 2025-12-31.

| Input | Endpoint | Model | Underlying source and licence |
|---|---|---|---|
| Day-ahead GFS (primary) | `previous-runs-api.open-meteo.com/v1/forecast`, variables with suffix `_previous_day1` | `gfs_seamless` | NOAA GFS and HRRR; US government work, public domain |
| Day-ahead ECMWF (robustness) | same | `ecmwf_ifs025` | ECMWF IFS open data, CC BY 4.0 |
| ERA5 (hindsight, not a forecast) | `archive-api.open-meteo.com/v1/archive` | `era5` | Copernicus Climate Change Service ERA5 (Hersbach et al. 2020, doi:10.1002/qj.3803); Copernicus licence, free reuse with attribution: contains modified Copernicus Climate Change Service information 2024-2025 |

The full query strings are in `data/manifest.json`. `_previous_day1` is the value the model issued 24 hours before the valid hour. That archive starts in January 2024, which sets the study window. Open-Meteo's Historical Forecast API is not used, because it stitches the first hours of each run and so describes a shorter lead time than a day-ahead forecaster has.

Open-Meteo responses carry a `generationtime_ms` field that changes on every request; the checksum covers the rest of the response. The archive is a third-party record of past runs and could change; the checksums would show it.
