from pathlib import Path
from scripts.package_release import REQUIRED, verify_release

def test_verify_release_detects_missing(tmp_path):
    (tmp_path / "genre.onnx").write_bytes(b"x")
    ok, missing = verify_release(tmp_path)
    assert not ok
    assert "thresholds.json" in missing

def test_verify_release_passes_when_complete(tmp_path):
    for name in REQUIRED:
        (tmp_path / name).write_text("x")
    ok, missing = verify_release(tmp_path)
    assert ok and missing == []
