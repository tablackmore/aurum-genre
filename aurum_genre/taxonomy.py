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

def output_labels(tax: dict) -> list[str]:
    """Full model output vocabulary: roots first, then namespaced subgenres
    (`electronic:techno`, ...) in taxonomy order. Drives the model's class count."""
    labels = list(tax["roots"])
    for root, subs in tax.get("sub", {}).items():
        labels.extend(f"{root}:{s}" for s in subs)
    return labels

def map_fma_subgenres(fma_titles: list[str], tax: dict) -> list[str]:
    """Map a track's fine FMA genre titles to namespaced subgenre labels
    (`electronic:<sub>`), deduped in first-seen order. Unknown titles are dropped."""
    sub_map = tax.get("fma_sub_map", {})
    # invert sub lists to find which root each subgenre belongs under
    root_of = {s: root for root, subs in tax.get("sub", {}).items() for s in subs}
    out: list[str] = []
    for title in fma_titles:
        sub = sub_map.get(title)
        if sub is None:
            continue
        label = f"{root_of.get(sub, 'electronic')}:{sub}"
        if label not in out:
            out.append(label)
    return out
