"""Test the mood manifest builder — keeps CC-BY target-mood tracks with audio."""
from __future__ import annotations
import textwrap
import pandas as pd
from scripts.build_mood_manifest import build


def test_keeps_ccby_target_moods_with_audio(tmp_path):
    meta = tmp_path / "meta"
    (meta / "data").mkdir(parents=True)
    (meta / "audio_licenses.txt").write_text(textwrap.dedent("""\
        10/100.mp3
        A by X from Jamendo: http://www.jamendo.com/track/100
        Available under a Creative Commons Attribution license: http://creativecommons.org/licenses/by/3.0/

        20/200.mp3
        B by Y from Jamendo: http://www.jamendo.com/track/200
        Available under a Creative Commons Attribution-NonCommercial license: http://creativecommons.org/licenses/by-nc/3.0/
    """))
    # track 100 = CC-BY, tagged happy+film; track 200 = CC-BY-NC (drop)
    (meta / "data" / "autotagging_moodtheme.tsv").write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "track_100\ta\tb\t10/100.mp3\t30\tmood/theme---happy\tmood/theme---film\n"
        "track_200\ta\tb\t20/200.mp3\t30\tmood/theme---sad\n"
    )
    # audio only for track 100
    audio = tmp_path / "audio" / "10"
    audio.mkdir(parents=True)
    (audio / "100.low.mp3").write_bytes(b"x")

    n_tr, n_va = build(meta, tmp_path / "audio",
                       tmp_path / "tr.csv", tmp_path / "va.csv",
                       moods=["happy", "sad"], val_frac=0.0)
    assert n_tr == 1 and n_va == 0                         # only track 100; kept in train
    rows = pd.read_csv(tmp_path / "tr.csv")
    # 'film' is not a target mood → dropped; only 'happy' kept
    assert rows.iloc[0]["root_labels"] == "happy"
    assert rows.iloc[0]["filepath"].endswith("10/100.low.mp3")
