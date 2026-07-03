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
    # Includes set.subset/set.split (present in real FMA metadata).
    content = textwrap.dedent("""\
        ,track,track,track,artist,set,set
        track_id,genre_top,license,title,name,subset,split
        1,Hip-Hop,Creative Commons Attribution,Song A,Artist A,small,training
        2,Electronic,Creative Commons Attribution,Song B,Artist B,medium,training
        3,Hip-Hop,Attribution-NonCommercial,Song C,Artist C,small,validation
        4,Classical,Creative Commons Attribution,Song D,Artist D,medium,test
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


def test_subset_and_split_filtering(tmp_path):
    """--subset restricts hierarchically; --split restricts to one split."""
    fma_meta = tmp_path / "fma_metadata"
    fma_meta.mkdir()
    _write_fake_tracks_csv(fma_meta / "tracks.csv")
    fma_audio = tmp_path / "fma_audio"
    fma_audio.mkdir()
    notice = tmp_path / "NOTICE"

    from scripts.build_manifest import build

    # subset=small keeps only small tracks; track 3 (small) is NC-filtered,
    # track 1 (small, CC-BY, Hip-Hop) survives; tracks 2 & 4 are medium → excluded.
    out_small = tmp_path / "small.csv"
    build(fma_meta, fma_audio, out_small, notice, subset="small")
    small = pd.read_csv(out_small)
    assert set(small["root_labels"]) == {"hip-hop"}
    assert len(small) == 1

    # split=training keeps tracks 1 & 2 (both permissive, training); 3=validation(NC), 4=test.
    out_train = tmp_path / "train.csv"
    build(fma_meta, fma_audio, out_train, notice, split="training")
    train = pd.read_csv(out_train)
    assert set(train["root_labels"]) == {"hip-hop", "electronic"}
    assert len(train) == 2


def test_electronic_track_gets_namespaced_subgenre_labels(tmp_path):
    """An electronic track whose fine genres include Techno gets electronic:techno;
    non-electronic tracks stay root-only; missing genres degrade gracefully."""
    fma_meta = tmp_path / "fma_metadata"
    fma_meta.mkdir()
    # tracks.csv WITH a track.genres column (list of fine genre IDs)
    content = textwrap.dedent("""\
        ,track,track,track,track,artist,set,set
        track_id,genre_top,genres,license,title,name,subset,split
        1,Electronic,[181],Creative Commons Attribution,Song A,Artist A,small,training
        2,Rock,[12],Creative Commons Attribution,Song B,Artist B,small,training
        3,Electronic,[38],Creative Commons Attribution,Song C,Artist C,small,training
    """)
    (fma_meta / "tracks.csv").write_text(content)
    # genres.csv: 181=Techno (Electronic subgenre), 38=Experimental (no sub), 12=Rock
    (fma_meta / "genres.csv").write_text(
        "genre_id,#tracks,parent,title,top_level\n"
        "181,100,15,Techno,15\n38,100,0,Experimental,38\n12,100,0,Rock,12\n"
    )
    fma_audio = tmp_path / "fma_audio"
    fma_audio.mkdir()
    out = tmp_path / "m.csv"
    notice = tmp_path / "NOTICE"

    from scripts.build_manifest import build
    build(fma_meta, fma_audio, out, notice, split="training")
    labels = set(pd.read_csv(out)["root_labels"])
    assert "electronic|electronic:techno" in labels   # track 1: root + subgenre
    assert "rock" in labels                            # track 2: root only
    assert "electronic" in labels                      # track 3: electronic, no known sub


def test_rock_track_gets_rock_subgenre_label(tmp_path):
    """Subgenres generalise beyond electronic: a Metal fine-genre → rock:metal."""
    fma_meta = tmp_path / "fma_metadata"
    fma_meta.mkdir()
    (fma_meta / "tracks.csv").write_text(textwrap.dedent("""\
        ,track,track,track,track,artist,set,set
        track_id,genre_top,genres,license,title,name,subset,split
        1,Rock,[45],Creative Commons Attribution,Song A,Artist A,small,training
    """))
    (fma_meta / "genres.csv").write_text(
        "genre_id,#tracks,parent,title,top_level\n45,50,12,Metal,12\n")
    fma_audio = tmp_path / "fma_audio"
    fma_audio.mkdir()
    from scripts.build_manifest import build
    build(fma_meta, fma_audio, tmp_path / "m.csv", tmp_path / "NOTICE", split="training")
    assert "rock|rock:metal" in set(pd.read_csv(tmp_path / "m.csv")["root_labels"])


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
