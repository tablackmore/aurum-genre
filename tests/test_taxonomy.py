from aurum_genre.taxonomy import load_taxonomy, root_labels, map_fma_root

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
