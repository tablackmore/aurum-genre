# SP3: Electronic-Subgenre Classifier

**Date:** 2026-07-03
**Status:** Design (decisions made autonomously per user delegation — review on return)
**Depends on:** SP1 (provenance machinery). Branch off `feat/reproducibility-provenance`.

## Context

The shipped genre-v1 tagger emits 11 root genres; every electronic subgenre (house,
techno, IDM, ambient, chiptune, …) collapses into a single `electronic` bucket. The user
asked for a classifier that distinguishes electronic subgenres. The data supports it: the
FMA-large permissive set alone has ~10 electronic subgenres with 40–234 tracks each, and
the incoming Jamendo CC-BY set adds more (ambient, trance, techno, house).

**Goal:** a genre-v2 model that outputs the 11 roots **plus** electronic subgenres, trained
reproducibly through SP1's pipeline, with a working FMA-only proof-of-concept tonight and a
clear path to strengthen sparse subgenres with Jamendo data.

## Decisions (made autonomously; rationale recorded for review)

**D1 — Single hierarchical multi-label model** (not a separate two-stage electronic-only
model). One `ShortChunkCNN` outputs roots + subgenres; a Techno track is labelled
`electronic` AND `electronic:techno` (multi-label BCE already supports this).
*Rationale:* maximal reuse — the whole pipeline (model, training, export, provenance) is
label-count-driven, so this needs no architecture change; the taxonomy already has a `sub`
field designed for exactly this; root behaviour is preserved as a superset; one model keeps
the on-device budget simple. Two-stage rejected: two models to ship + a routing threshold.

**D2 — Namespaced subgenre labels.** Subgenre labels are `electronic:<sub>` in the flat
output vocabulary. Output label list = `roots + ["electronic:"+s for s in sub.electronic]`.
Order is deterministic (roots first, then subgenres in taxonomy order).

**D3 — Subgenres attach only to electronic tracks.** Non-electronic tracks keep their root
labels only. Electronic tracks get `electronic` + any matched subgenre(s) (multi-label; a
track may carry several subgenre tags).

**D4 — Subgenre vocabulary (forward-looking, 12).**
`ambient, chiptune, dance, downtempo, drum-n-bass, dubstep, house, idm, minimal, techno,
trance, trip-hop`. FMA populates 10 of these well; `trance` and `drum-n-bass` stay sparse
until Jamendo lands (they'll show as low/'n/a' support in metrics — honest, and no re-work
needed when Jamendo arrives).

**D5 — FMA subgenre extraction.** A track's fine genres live in `tracks.csv` `("track",
"genres")` as a list of genre IDs. `build_manifest` resolves those IDs to titles via
`fma_metadata/genres.csv`, maps Electronic-descendant titles to subgenre labels through a
`fma_sub_map`, and appends `electronic:<sub>` labels. The generic "Electronic" tag maps to
no subgenre (root only).

**D6 — Keep checkpoint key `roots` = full output label list.** To avoid churn/risk across
`export.py`/`eval.py`/`infer.py` (all read `blob["roots"]` and use `len(...)`), the
checkpoint's `roots` key now holds the complete output-label list (roots + subgenres). It is
functionally the model's class list; documented as such. (A future rename to `labels` is
a clean follow-up, out of scope tonight.)

**D7 — Scope tonight = FMA-only genre-v2 proof-of-concept.** Train on FMA-large now; produce
per-subgenre metrics + a run_manifest. Merging Jamendo CC-BY data (SP2) to strengthen sparse
subgenres is a follow-on once the download + license-filter complete.

## Taxonomy changes (`taxonomy.json`)

- `version`: `genre-v2`
- `sub`: `{"electronic": [the 12 in D4]}`
- add `fma_sub_map` (FMA fine-genre title → subgenre label):
  `Chip Music→chiptune, Chiptune→chiptune, IDM→idm, Techno→techno, Downtempo→downtempo,
  Ambient Electronic→ambient, House→house, Trip-Hop→trip-hop, Minimal Electronic→minimal,
  Dubstep→dubstep, Dance→dance, Drum & Bass→drum-n-bass, Trance→trance`

## Code changes

| File | Change |
|---|---|
| `taxonomy.json` | version, `sub.electronic`, `fma_sub_map` |
| `aurum_genre/taxonomy.py` | `output_labels(tax)` → roots + namespaced subs; `map_fma_subgenres(titles, tax)` → list of `electronic:<sub>` |
| `scripts/build_manifest.py` | load `genres.csv`; for each track resolve fine-genre IDs→titles→subgenre labels; append to `root_labels` column |
| `aurum_genre/train.py`, `aurum_genre/eval.py`, `scripts/run_pipeline.py` | use `output_labels` instead of `root_labels` for the model's class list / pos_weight / eval |

Model, mel, export, provenance, seeding: **unchanged** (all label-count-driven). The SP1
`run_manifest.json` now records the larger label set and per-subgenre metrics automatically.

## Testing

- `taxonomy`: `output_labels` returns roots then `electronic:*`; `map_fma_subgenres`
  maps known titles and drops unknowns; the generic "Electronic" title yields no subgenre.
- `build_manifest`: an electronic track with a Techno fine-genre emits
  `electronic|electronic:techno`; a rock track emits only `rock`.
- Existing suite stays green (root-only manifests still valid — subgenre columns are additive).

## Verification

1. `pytest` green.
2. `build_manifest --subset large` produces train/val manifests whose electronic rows carry
   `electronic:*` labels; count per subgenre matches the FMA analysis (chiptune ~234, idm
   ~148, …).
3. `run_pipeline --subset large` trains genre-v2, and `release/run_manifest.json` shows the
   expanded label set with per-subgenre AUC + support (sparse trance/drum-n-bass flagged).
4. Report per-subgenre results; note which need Jamendo data.

## Out of scope tonight
- Jamendo data merge (SP2) — strengthens sparse subgenres; separate work once download lands.
- Subgenres for non-electronic roots.
- Renaming checkpoint `roots`→`labels`.
