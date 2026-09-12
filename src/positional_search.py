"""
Part C: exact phrase search and ordered-proximity ("WITHIN/k") search,
built directly on the positional index -- NOT merely on doc-level
term co-occurrence. This is the key conceptual difference the
assignment calls out between VSM (Part B) and positional retrieval
(Part C): VSM only knows a term occurred in a document, the positional
index additionally knows WHERE, so we can require terms to occur in
a specific order and within a bounded gap.
"""

from src.preprocess import preprocess


def candidate_docs(positional_index, terms):
    """Documents that contain ALL terms (necessary condition before
    checking positions)."""
    if not terms:
        return set()
    sets = []
    for t in terms:
        entry = positional_index.get(t)
        if not entry:
            return set()
        sets.append(set(entry["postings"].keys()))
    return set.intersection(*sets)


def phrase_search(positional_index, phrase_text):
    """
    Exact phrase search: terms must appear CONSECUTIVELY (gap of exactly
    1 stemmed-token position) and in the given order in the document.

    Returns: dict docid -> list of starting positions where the phrase matches
    """
    terms = preprocess(phrase_text)
    if len(terms) < 2:
        # degrade gracefully to a single-term lookup
        entry = positional_index.get(terms[0]) if terms else None
        if not entry:
            return {}
        return {d: p["positions"] for d, p in entry["postings"].items()}

    docs = candidate_docs(positional_index, terms)
    matches = {}
    for docid in docs:
        # positions of the first term
        pos_lists = [positional_index[t]["postings"][docid]["positions"] for t in terms]
        first_positions = set(pos_lists[0])
        hits = []
        for start in first_positions:
            ok = True
            for offset, plist in enumerate(pos_lists[1:], start=1):
                if (start + offset) not in plist:
                    ok = False
                    break
            if ok:
                hits.append(start)
        if hits:
            matches[docid] = sorted(hits)
    return matches


def proximity_search(positional_index, term_a, term_b, k, ordered=True):
    """
    Ordered proximity ("term_a WITHIN/k term_b"): term_b must occur
    within k token positions AFTER term_a (1 <= gap <= k).
    If ordered=False, checks |pos_b - pos_a| <= k in either direction.

    Returns: dict docid -> list of (pos_a, pos_b) satisfying pairs.
    """
    ta = preprocess(term_a)
    tb = preprocess(term_b)
    if not ta or not tb:
        return {}
    ta, tb = ta[0], tb[0]

    docs = candidate_docs(positional_index, [ta, tb])
    matches = {}
    for docid in docs:
        pos_a = positional_index[ta]["postings"][docid]["positions"]
        pos_b = positional_index[tb]["postings"][docid]["positions"]
        hits = []
        for pa in pos_a:
            for pb in pos_b:
                gap = pb - pa
                if ordered:
                    if 0 < gap <= k:
                        hits.append((pa, pb))
                else:
                    if 0 < abs(gap) <= k:
                        hits.append((pa, pb))
        if hits:
            matches[docid] = sorted(set(hits))
    return matches
