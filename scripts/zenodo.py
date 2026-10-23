"""Archive the working paper and its code on Zenodo, with a DOI printed on the paper itself.

    python scripts/zenodo.py reserve   # draft record + reserved DOI; writes paper/doi.tex
    (rebuild the paper, commit)
    python scripts/zenodo.py upload    # PDF and a source archive of HEAD into the draft
    python scripts/zenodo.py publish   # makes the record public and registers the DOI
    python scripts/zenodo.py status

The token comes from ZENODO_TOKEN or the file ~/.zenodo-token (scopes deposit:write and
deposit:actions). It is never written anywhere. The draft's id is kept in zenodo/deposit.json
(ignored by git). Uses only the standard library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://zenodo.org/api/deposit/depositions"
STATE = ROOT / "zenodo" / "deposit.json"
PDF = ROOT / "paper" / "solar-two-sites-v1.pdf"
VERSION = "1.0"

METADATA = {
    "upload_type": "publication",
    "publication_type": "workingpaper",
    "title": "How much of a solar forecast is skill? A two-site day-ahead study with public data",
    "creators": [{"name": "Peyman, Saeed", "affiliation": "Independent researcher"}],
    "publication_date": "2026-10-23",
    "version": VERSION,
    "language": "eng",
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": [
        "solar forecasting",
        "photovoltaics",
        "day-ahead forecasting",
        "numerical weather prediction",
        "gradient boosting",
        "forecast verification",
        "reproducibility",
    ],
    "notes": (
        "Working paper, version 1.0, not peer reviewed. The paper and figures are CC BY 4.0; "
        "the code in the source archive is MIT-licensed. Data sources and licences are listed "
        "in DATA.md inside the archive."
    ),
    "related_identifiers": [
        {
            "identifier": "https://github.com/speyyman/solar-two-sites",
            "relation": "isSupplementedBy",
            "resource_type": "software",
        },
        {
            "identifier": "https://speyyman.github.io/research/solar-two-sites/",
            "relation": "isIdenticalTo",
        },
    ],
}


def token() -> str:
    t = os.environ.get("ZENODO_TOKEN")
    f = Path.home() / ".zenodo-token"
    if not t and f.exists():
        t = f.read_text(encoding="utf8").strip()
    if not t:
        sys.exit("No token: set ZENODO_TOKEN or put it in ~/.zenodo-token")
    return t


def call(method: str, url: str, body: bytes | None = None, ctype: str = "application/json") -> dict:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token()}")
    if body is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> {e.code}: {e.read().decode(errors='replace')}")
    return json.loads(raw) if raw else {}


def description() -> str:
    abstract = (ROOT / "zenodo" / "abstract.txt").read_text(encoding="utf8").strip()
    return f"<p>{abstract}</p>"


def load() -> dict:
    if not STATE.exists():
        sys.exit("No draft yet: run `reserve` first")
    return json.loads(STATE.read_text(encoding="utf8"))


def reserve() -> None:
    if STATE.exists():
        sys.exit(f"A draft already exists: {STATE}")
    meta = {**METADATA, "description": description(), "prereserve_doi": True}
    dep = call("POST", API, json.dumps({"metadata": meta}).encode())
    doi = dep["metadata"]["prereserve_doi"]["doi"]
    STATE.write_text(
        json.dumps({"id": dep["id"], "doi": doi, "bucket": dep["links"]["bucket"]}, indent=2),
        encoding="utf8",
    )
    (ROOT / "paper" / "doi.tex").write_text(
        "% The Zenodo DOI for this version (scripts/zenodo.py writes it). Empty until reserved.\n"
        f"\\def\\paperdoi{{{doi}}}\n",
        encoding="utf8",
    )
    print(f"Draft {dep['id']} reserved DOI {doi}. Rebuild the paper, then run `upload`.")


def upload() -> None:
    st = load()
    if st["doi"] not in (ROOT / "paper" / "doi.tex").read_text(encoding="utf8"):
        sys.exit("paper/doi.tex does not carry the reserved DOI")
    files = {PDF.name: PDF.read_bytes()}
    with tempfile.TemporaryDirectory() as d:
        zip_path = Path(d) / f"solar-two-sites-{VERSION}-source.zip"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=zip",
                f"--prefix=solar-two-sites-{VERSION}/",
                "-o",
                str(zip_path),
                "HEAD",
            ],
            cwd=ROOT,
            check=True,
        )
        files[zip_path.name] = zip_path.read_bytes()
    for name, data in files.items():
        r = call("PUT", f"{st['bucket']}/{name}", data, "application/octet-stream")
        print(f"uploaded {name}: {r.get('size')} bytes, {r.get('checksum')}")


def publish() -> None:
    st = load()
    dep = call("POST", f"{API}/{st['id']}/actions/publish")
    print(f"Published: https://doi.org/{dep.get('doi', st['doi'])}  ({dep['links'].get('html')})")


def status() -> None:
    st = load()
    dep = call("GET", f"{API}/{st['id']}")
    print(json.dumps({k: dep.get(k) for k in ("id", "state", "submitted", "doi")}, indent=2))
    for f in dep.get("files", []):
        print(f"  {f['filename']}  {f['filesize']} bytes")


if __name__ == "__main__":
    cmds = {"reserve": reserve, "upload": upload, "publish": publish, "status": status}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
