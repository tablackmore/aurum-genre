from aurum_genre.taxonomy import (load_taxonomy, root_labels, map_fma_root,
                                  output_labels, map_fma_subgenres)

def test_root_labels_are_ordered_and_unique():
    tax = load_taxonomy("taxonomy.json")
    labels = root_labels(tax)
    assert labels == list(dict.fromkeys(labels))  # unique, order-preserving
    assert "house" in labels or "electronic" in labels

def test_every_fma_root_maps_or_is_explicitly_dropped():
    tax = load_taxonomy("taxonomy.json")
    fma_roots = ["Electronic", "Rock", "Hip-Hop", "Pop", "Jazz", "Classical",
                 "Folk", "Soul-RnB", "Country", "Blues", "International",
                 "Experimental", "Instrumental", "Easy Listening",
                 "Old-Time / Historic", "Spoken"]
    for g in fma_roots:
        # must not raise KeyError; None means intentionally dropped
        _ = map_fma_root(g, tax)  # totality: mapping is defined for all inputs
    assert map_fma_root("Spoken", tax) is None       # dropped
    assert map_fma_root("Electronic", tax) is not None

def test_output_labels_are_roots_then_namespaced_subgenres():
    tax = load_taxonomy("taxonomy.json")
    labels = output_labels(tax)
    roots = root_labels(tax)
    assert labels[:len(roots)] == roots               # roots first, in order
    subs = labels[len(roots):]
    assert all(s.startswith("electronic:") for s in subs)
    assert "electronic:techno" in subs and "electronic:chiptune" in subs
    assert labels == list(dict.fromkeys(labels))      # unique

def test_map_fma_subgenres_maps_known_titles_and_drops_unknown():
    tax = load_taxonomy("taxonomy.json")
    # A track tagged Techno + Chip Music → two namespaced subgenre labels.
    got = map_fma_subgenres(["Techno", "Chip Music"], tax)
    assert set(got) == {"electronic:techno", "electronic:chiptune"}
    # The generic "Electronic" title is not a subgenre; unknown titles are dropped.
    assert map_fma_subgenres(["Electronic", "Nonsense Genre"], tax) == []
