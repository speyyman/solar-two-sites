import json

import pytest

from pipeline import fetch


@pytest.fixture
def local_plan(tmp_path, monkeypatch):
    data = tmp_path / "data"
    f = data / "raw" / "x.csv"
    f.parent.mkdir(parents=True)
    f.write_text("a,b\n1,2\n")
    monkeypatch.setattr(fetch, "DATA", data)
    monkeypatch.setattr(fetch, "MANIFEST", data / "manifest.json")
    monkeypatch.setattr(fetch, "plan", lambda: [("https://example.invalid/x.csv", f)])
    monkeypatch.delenv("UPDATE_MANIFEST", raising=False)
    return data, f


def test_matching_file_passes_and_writes_manifest(local_plan):
    data, f = local_plan
    m = fetch.fetch()
    assert m["raw/x.csv"]["sha256"] == fetch.sha256(f)
    assert json.loads((data / "manifest.json").read_text()) == m


def test_changed_file_stops_the_run(local_plan):
    data, f = local_plan
    fetch.fetch()
    before = (data / "manifest.json").read_text()
    f.write_text("a,b\n1,3\n")
    with pytest.raises(SystemExit, match="differ from data/manifest.json"):
        fetch.fetch()
    assert (data / "manifest.json").read_text() == before


def test_changed_file_accepted_on_purpose(local_plan):
    data, f = local_plan
    fetch.fetch()
    f.write_text("a,b\n1,3\n")
    m = fetch.fetch(update=True)
    assert m["raw/x.csv"]["sha256"] == fetch.sha256(f)
