import pandas as pd
from aurum_genre.licenses import is_permissive, filter_permissive, build_notice

def test_permissive_classification():
    assert is_permissive("Attribution 4.0 International (CC BY 4.0)")
    assert is_permissive("Creative Commons Attribution")
    assert is_permissive("CC0 1.0 Universal")
    assert is_permissive("Public Domain Mark 1.0")
    # DISQUALIFYING:
    assert not is_permissive("Attribution-NonCommercial 4.0")
    assert not is_permissive("Attribution-NoDerivatives 4.0")
    assert not is_permissive("Attribution-NonCommercial-ShareAlike 3.0")
    assert not is_permissive("All rights reserved")
    assert not is_permissive("")
    assert not is_permissive(None)

def test_filter_leaks_zero_nonpermissive():
    df = pd.DataFrame({
        "track_id": [1, 2, 3, 4],
        "license": [
            "Attribution 4.0 International (CC BY 4.0)",
            "Attribution-NonCommercial 4.0",   # must be dropped
            "CC0 1.0 Universal",
            "Attribution-NoDerivatives 4.0",   # must be dropped
        ],
    })
    out = filter_permissive(df, "license")
    assert set(out["track_id"]) == {1, 3}
    assert all(is_permissive(l) for l in out["license"])

def test_notice_credits_ccby_tracks():
    df = pd.DataFrame({
        "track_id": [1],
        "artist_name": ["Some Artist"],
        "track_title": ["A Song"],
        "license": ["Attribution 4.0 International (CC BY 4.0)"],
    })
    notice = build_notice(df)
    assert "Some Artist" in notice and "A Song" in notice and "CC BY" in notice
