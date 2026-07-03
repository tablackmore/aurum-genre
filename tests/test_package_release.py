from scripts.package_release import verify_release, REQUIRED

def test_run_manifest_is_required(tmp_path):
    assert "run_manifest.json" in REQUIRED
    for name in REQUIRED:
        if name != "run_manifest.json":
            (tmp_path / name).write_text("x")
    ok, missing = verify_release(tmp_path)
    assert not ok and "run_manifest.json" in missing
