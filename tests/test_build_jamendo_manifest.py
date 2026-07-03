"""Tests for the MTG-Jamendo CC-BY manifest builder."""
from __future__ import annotations
import textwrap
import pandas as pd
from scripts.build_jamendo_manifest import build, labels_for_tags

GMAP = {
    "root_map": {"techno": "electronic", "electronic": "electronic",
                 "rock": "rock", "metal": "rock"},
    "sub_map": {"techno": "techno", "metal": "metal"},
}


def test_labels_for_tags_namespaces_subgenre_under_its_root():
    assert labels_for_tags(["techno"], GMAP) == ["electronic", "electronic:techno"]
    assert labels_for_tags(["metal"], GMAP) == ["rock", "rock:metal"]   # generalised
    assert labels_for_tags(["rock"], GMAP) == ["rock"]
    assert labels_for_tags(["unknown"], GMAP) == []


def test_build_keeps_ccby_and_maps_labels(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    # audio_licenses.txt: track 100 is CC-BY (kept), track 200 is CC-BY-NC (dropped)
    (meta / "audio_licenses.txt").write_text(textwrap.dedent("""\
        10/100.mp3
        Song A by Artist from Jamendo: http://www.jamendo.com/track/100
        Available under a Creative Commons Attribution license: http://creativecommons.org/licenses/by/3.0/

        20/200.mp3
        Song B by Artist from Jamendo: http://www.jamendo.com/track/200
        Available under a Creative Commons Attribution-NonCommercial license: http://creativecommons.org/licenses/by-nc/3.0/
    """))
    # autotagging_genre.tsv: track 100 tagged techno; track 200 tagged rock
    (meta / "autotagging_genre.tsv").write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "track_100\tartist_1\talbum_1\t10/100.mp3\t30.0\tgenre---techno\n"
        "track_200\tartist_2\talbum_2\t20/200.mp3\t30.0\tgenre---rock\n"
    )
    gmap = tmp_path / "map.json"
    import json
    gmap.write_text(json.dumps(GMAP))

    out = tmp_path / "jamendo.csv"
    n = build(meta, "data/mtg_jamendo", out, gmap, audio_ext=".low.mp3")
    assert n == 1                                   # only the CC-BY track survives
    df = pd.read_csv(out)
    assert df.iloc[0]["root_labels"] == "electronic|electronic:techno"
    assert df.iloc[0]["filepath"].endswith("10/100.low.mp3")
