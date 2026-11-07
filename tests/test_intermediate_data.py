"""Integration checks on the real processed data (skipped if not yet built)."""
import pytest

from predictops.config import load_config, project_path
from predictops.data.ingestion import load_intermediate

CFG = load_config()
PARQUET = project_path(CFG["paths"]["intermediate_dir"]) / "metropt3.parquet"
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="run scripts/build_intermediate.py first")


def test_row_count_matches_raw():
    assert len(load_intermediate(columns=["timestamp"])) == 1_516_948


def test_sorted_and_unique():
    t = load_intermediate(columns=["timestamp"]).timestamp
    assert t.is_monotonic_increasing and not t.duplicated().any()


def test_raw_file_untouched():
    import hashlib
    raw = project_path(CFG["paths"]["raw_csv"])
    expected = (raw.parent / "SHA256SUMS").read_text().split("\n")[1].split()[0]
    h = hashlib.sha256()
    with open(raw, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == expected
