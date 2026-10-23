"""Download every raw input and record a checksum for each file.

Raw files go to data/raw/ (not committed). data/manifest.json (committed) holds
the URL and a SHA-256 of each file's content, so a rerun can prove it saw the
same data. Weather requests are pinned to fixed dates, so they are repeatable
for as long as Open-Meteo keeps the archive unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
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
NL = "\n"


def sha256(path: Path) -> str:
    """Checksum of the file's content.

    Open-Meteo responses carry the server's `generationtime_ms`, which changes on every
    request, so for JSON the checksum covers everything except that field.
    """
    h = hashlib.sha256()
    if path.suffix == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            doc.pop("generationtime_ms", None)
        h.update(json.dumps(doc, sort_keys=True).encode("utf-8"))
        return h.hexdigest()
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


def fetch(force: bool = False, update: bool | None = None) -> dict[str, dict[str, object]]:
    """Download what is missing and check every file against the committed manifest.

    A file whose checksum (or URL) differs from the manifest stops the run, because every
    result downstream would silently change. To accept new data on purpose, set the
    environment variable UPDATE_MANIFEST=1; the manifest is then rewritten and
    `git diff data/manifest.json` shows what changed.
    """
    if update is None:
        update = os.environ.get("UPDATE_MANIFEST") == "1"
    manifest: dict[str, dict[str, object]] = {}
    for url, dest in plan():
        if force or not dest.exists():
            print(f"fetch {dest.relative_to(DATA)}")
            download(url, dest)
        rel = dest.relative_to(DATA).as_posix()
        manifest[rel] = {"url": url, "sha256": sha256(dest)}
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    changed = [k for k, v in manifest.items() if k in previous and previous[k] != v]
    if changed and not update:
        raise SystemExit(
            "these inputs differ from data/manifest.json: "
            + ", ".join(changed)
            + NL
            + "rerun with UPDATE_MANIFEST=1 to accept them"
        )
    for k in changed:
        print(f"updated manifest entry: {k}")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + NL, newline=NL)
    return manifest
