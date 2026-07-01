"""Load the genre taxonomy and map FMA root genres onto AURUM root labels.

`fma_root_map` is authoritative; a genre listed in `dropped` (or absent from
both) maps to None (intentionally excluded from training)."""
from __future__ import annotations
import json
from pathlib import Path

def load_taxonomy(path: str | Path = "taxonomy.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def root_labels(tax: dict) -> list[str]:
    return list(tax["roots"])

def map_fma_root(fma_genre: str, tax: dict) -> str | None:
    if fma_genre in tax.get("dropped", []):
        return None
    return tax.get("fma_root_map", {}).get(fma_genre, None)
