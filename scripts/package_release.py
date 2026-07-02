"""Assemble + verify the genre-v1 release asset directory."""
from __future__ import annotations
from pathlib import Path

REQUIRED = ["genre.onnx", "taxonomy.json", "thresholds.json",
            "mel_recipe.txt", "mel_golden.npz", "NOTICE", "license_manifest.csv",
            "run_manifest.json"]

def verify_release(release_dir) -> tuple[bool, list[str]]:
    d = Path(release_dir)
    missing = [n for n in REQUIRED if not (d / n).is_file()]
    return (len(missing) == 0, missing)

def write_manifest(release_dir) -> None:
    d = Path(release_dir)
    (d / "MANIFEST.txt").write_text("\n".join(sorted(p.name for p in d.iterdir())) + "\n")

if __name__ == "__main__":
    import sys
    ok, missing = verify_release(sys.argv[1])
    if not ok:
        raise SystemExit(f"release incomplete, missing: {missing}")
    write_manifest(sys.argv[1])
    print("release OK")
