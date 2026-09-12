"""
NOVELTY FEATURE 4 -- typo-tolerant ("did you mean") query term correction.

Motivation: Part E explicitly requires testing "a query containing a
term that does not occur in the corpus" -- normally that just returns
zero/near-zero results. Real product-search engines instead treat an
out-of-vocabulary query term as a likely typo and fall back to the
closest in-vocabulary term. This robustness problem is well studied in
e-commerce IR (e.g. Duan & Hsu, "Online Spelling Correction for Query
Completion", WWW 2011; more recently neural/typo-tolerant retrievers
such as Hong et al., "Typo-Aware Dense Retrieval", ACL Findings 2023,
motivate the same failure mode -- OOV/misspelled query terms silently
losing recall). We implement the classical, fully offline baseline:
minimum edit (Levenshtein) distance between the OOV stemmed query term
and every term in our dictionary, substituting the closest match when
it is "close enough" (distance <= 2 and short relative length ratio).
"""

from src.preprocess import preprocess


def levenshtein(a, b):
    """Compute the minimum number of single-character edits needed to change one word into another.

    This is the classic edit-distance metric. We use it here to measure how close an
    out-of-vocabulary query term is to a valid corpus term. A small distance means the
    user likely mistyped the word, while a large distance suggests the term is genuinely
    unrelated to the indexed vocabulary.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Dynamic-programming table: prev[j] is the distance for the previous prefix of `a`.
    # We update each cell using the standard Levenshtein recurrence.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def suggest_correction(term, dictionary_terms, max_distance=2):
    """Pick the closest valid dictionary term for a misspelled input.

    We scan through all indexed vocabulary terms and find the candidate with the
    smallest edit distance. To keep the search lightweight, we first reject words
    whose length differs too much from the query term. If the best candidate is
    within `max_distance`, we treat it as a likely typo correction; otherwise we
    return None and leave the word untouched.
    """
    best, best_dist = None, max_distance + 1
    for cand in dictionary_terms:
        # Cheap pre-check: if candidate length differs by more than the allowed distance,
        # it cannot be a close match and does not need a full dynamic-programming check.
        if abs(len(cand) - len(term)) > max_distance:
            continue

        d = levenshtein(term, cand)
        if d < best_dist or (d == best_dist and (best is None or cand < best)):
            best, best_dist = cand, d

    return best if best_dist <= max_distance else None


def correct_query(query_text, index):
    """Normalize a query by fixing likely typos in out-of-vocabulary terms.

    The pipeline is simple:
      1. tokenize the query
      2. keep terms already in the index as-is
      3. for unknown terms, search for the nearest valid token using edit distance
      4. replace only if the match is close enough; otherwise leave it unchanged

    This helps retrieval remain useful when a user types a misspelling such as
    'colr' instead of 'color', without breaking the overall search pipeline.
    """
    tokens = preprocess(query_text)
    dictionary = list(index.keys())
    corrections = {}
    corrected_tokens = []

    for t in tokens:
        if t in index:
            # Already valid: no need to do anything.
            corrected_tokens.append(t)
        else:
            # Unknown term: try to recover the nearest valid token.
            sug = suggest_correction(t, dictionary)
            if sug:
                corrections[t] = sug
                corrected_tokens.append(sug)
            else:
                # No close match found; keep the original token so we still have the user intent.
                corrected_tokens.append(t)

    return corrected_tokens, corrections
