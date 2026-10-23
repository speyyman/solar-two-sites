"""Download every raw input and record a checksum for each file.

Raw files go to data/raw/ (not committed). data/manifest.json (committed) holds
the URL, byte size and SHA-256 of each file, so a rerun can prove it saw the
same bytes. Weather requests are pinned to fixed dates, so they are repeatable
for as long as Open-Meteo keeps the archive unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from pipeline.config import (
    ARCHIVE_API,
    DATA,
    METADATA_FILES,
    NWP_MODELS,
    OEDI,
    PREVIOUS_RUNS_API,
    RAW,
    SITES,
    WEATHER_VARS,
    Site,
)

MANIFEST = DATA / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def weather_urls(site: Site) -> dict[str, str]:
    """Day-ahead forecasts (one per NWP model) and the ERA5 reanalysis."""
    # Ask for a few days either side so local days at the edges are complete.
    start, end = "2024-01-01", "2025-12-31"
    common = {"latitude": site.latitude, "longitude": site.longitude, "timezone": "UTC"}
    urls = {}
    for key, model in NWP_MODELS.items():
        q = dict(
            common,
            start_date=start,
            end_date=end,
            models=model,
            hourly=",".join(f"{v}_previous_day1" for v in WEATHER_VARS),
        )
        urls[f"nwp_{key}.json"] = f"{PREVIOUS_RUNS_API}?{urlencode(q)}"
    q = dict(common, start_date=start, end_date=end, models="era5", hourly=",".join(WEATHER_VARS))
    urls["era5.json"] = f"{ARCHIVE_API}?{urlencode(q)}"
    return urls


def plan() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for key, site in SITES.items():
        d = RAW / key
        for f in (*site.pv_files, *site.irradiance_files):
            items.append((f"{OEDI}/{f.key}", d / Path(f.key).name))
        items.append((f"{OEDI}/{METADATA_FILES[key]}", d / Path(METADATA_FILES[key]).name))
        for name, url in weather_urls(site).items():
            items.append((url, d / name))
    return items


def fetch(force: bool = False) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for url, dest in plan():
        if force or not dest.exists():
            print(f"fetch {dest.relative_to(DATA)}")
            download(url, dest)
        rel = dest.relative_to(DATA).as_posix()
        manifest[rel] = {"url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)}
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    changed = [k for k, v in manifest.items() if k in previous and previous[k] != v]
    for k in changed:
        print(f"note: {k} differs from the committed manifest")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
