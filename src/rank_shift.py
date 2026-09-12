"""
NOVELTY -- backend port of swatchbook.html's rank-shift badges: how many
places a document moved under positional retrieval (Part C: phrase /
proximity) compared to where plain VSM lnc.ltc (Part B) would have
ranked the same query text. This is a direct, visual answer to Part E's
"explain at least two cases where positional information changes the
result set/order" requirement -- one badge per result row, computed
live from whatever the corpus actually returns (never hard-coded).
"""


def compute_baseline_rank_map(vsm_index, query_text, pool_size=100):
    """Rank position (1-indexed) of every document plain VSM returns for
    query_text, out to pool_size. Docs plain VSM never scores at all
    (e.g. every query term has idf=0) simply won't be keys in this map."""
    ranked, _ = vsm_index.search(query_text, top_k=pool_size)
    return {docid: i + 1 for i, (docid, _) in enumerate(ranked)}


def rank_shift_badge(docid, current_rank, baseline_map):
    """
    Returns a short badge string:
      "NEW"   -- not present in plain VSM's ranked results at all
      "="     -- same rank position under both
      "^N"    -- moved up N places under positional retrieval (better)
      "vN"    -- moved down N places under positional retrieval (worse)
    (Terminal-safe ASCII; the web UI shows the same information as
    unicode up/down triangles.)
    """
    if docid not in baseline_map:
        return "NEW"
    base_rank = baseline_map[docid]
    delta = base_rank - current_rank  # positive = moved up (better)
    if delta == 0:
        return "="
    return f"^{delta}" if delta > 0 else f"v{abs(delta)}"
