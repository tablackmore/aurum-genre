"""Tests for scripts/build_manifest.py — focused on license_manifest.csv output."""
from __future__ import annotations
import textwrap
from pathlib import Path
import pandas as pd
import pytest
from aurum_genre.licenses import is_permissive


def _write_fake_tracks_csv(path: Path) -> None:
    """Write a minimal multi-index tracks.csv that build() can parse."""
    # FMA tracks.csv has a two-row header: level-0 then level-1 column names.
    content = textwrap.dedent("""\
        ,track,track,track,artist
        track_id,genre_top,license,title,name
        1,Hip-Hop,Creative Commons Attribution,Song A,Artist A
        2,Electronic,Creative Commons Attribution,Song B,Artist B
        3,Hip-Hop,Attribution-NonCommercial,Song C,Artist C
        4,Classical,Creative Commons Attribution,Song D,Artist D
    """)
    path.write_text(content)


def test_license_manifest_exists_and_has_correct_columns(tmp_path):
    # Minimal fixture: build() needs fma_meta/tracks.csv + fma_audio dir.
    fma_meta = tmp_path / "fma_metadata"
    fma_meta.mkdir()
    _write_fake_tracks_csv(fma_meta / "tracks.csv")
    fma_audio = tmp_path / "fma_audio"
    fma_audio.mkdir()

    out = tmp_path / "manifest.csv"
    notice = tmp_path / "NOTICE"
    lic_manifest = tmp_path / "release" / "license_manifest.csv"

    # Import here so sys.path issues surface clearly.
    from scripts.build_manifest import build
    build(fma_meta, fma_audio, out, notice, lic_manifest)

    assert lic_manifest.exists(), "license_manifest.csv was not created"
    df = pd.read_csv(lic_manifest)

    required_cols = {"track_id", "artist_name", "track_title", "license", "root"}
    assert required_cols.issubset(df.columns), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )


def test_license_manifest_contains_only_permissive_licenses(tmp_path):
    fma_meta = tmp_path / "fma_metadata"
    fma_meta.mkdir()
    _write_fake_tracks_csv(fma_meta / "tracks.csv")
    fma_audio = tmp_path / "fma_audio"
    fma_audio.mkdir()

    out = tmp_path / "manifest.csv"
    notice = tmp_path / "NOTICE"
    lic_manifest = tmp_path / "license_manifest.csv"

    from scripts.build_manifest import build
    build(fma_meta, fma_audio, out, notice, lic_manifest)

    df = pd.read_csv(lic_manifest)
    assert len(df) > 0, "Expected at least one permissive track in fixture"
    for _, row in df.iterrows():
        assert is_permissive(row["license"]), (
            f"Non-permissive license found in manifest: {row['license']!r}"
        )


def test_license_manifest_not_written_when_arg_is_none(tmp_path):
    """When license_manifest=None, no file should be created."""
    fma_meta = tmp_path / "fma_metadata"
    fma_meta.mkdir()
    _write_fake_tracks_csv(fma_meta / "tracks.csv")
    fma_audio = tmp_path / "fma_audio"
    fma_audio.mkdir()

    out = tmp_path / "manifest.csv"
    notice = tmp_path / "NOTICE"

    from scripts.build_manifest import build
    build(fma_meta, fma_audio, out, notice, None)

    # No license_manifest.csv should exist anywhere under tmp_path (other than manifest.csv)
    created = [p for p in tmp_path.rglob("license_manifest.csv")]
    assert created == [], f"Unexpected file(s) created: {created}"
