"""Classify FMA per-track license strings and keep only permissive tracks.

Permissive (commercial-OK): CC-BY (any version), CC0, public domain.
DISQUALIFYING: NonCommercial (NC), NoDerivatives (ND), ShareAlike-only,
all-rights-reserved, unknown/empty.
"""
from __future__ import annotations
import pandas as pd

_DISQUALIFY = ("noncommercial", "non-commercial", "noderiv", "no-deriv",
               "sharealike", "share-alike", "-nc", "-nd", "-sa")
_PERMIT = ("cc by", "cc-by", "attribution", "cc0", "public domain", "publicdomain")

def is_permissive(license_str: str | None) -> bool:
    if not license_str:
        return False
    s = license_str.strip().lower()
    if any(bad in s for bad in _DISQUALIFY):
        return False
    return any(good in s for good in _PERMIT)

def filter_permissive(df: pd.DataFrame, license_col: str) -> pd.DataFrame:
    mask = df[license_col].apply(is_permissive)
    return df[mask].copy()

def build_notice(df: pd.DataFrame) -> str:
    lines = [
        "AURUM genre model — training data attribution (CC BY).",
        "Trained on Free Music Archive tracks under CC-BY/CC0/public-domain.",
        "",
    ]
    for _, r in df.iterrows():
        artist = r.get("artist_name", "Unknown artist")
        title = r.get("track_title", "Untitled")
        lic = r.get("license", "")
        lines.append(f"- {artist} — {title} ({lic})")
    return "\n".join(lines) + "\n"
