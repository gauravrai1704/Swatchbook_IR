"""
Part E: Testing.
  - >= 10 free-text queries (run through VSM, and also BM25/SDM for comparison)
  - >= 5 exact phrase queries
  - >= 3 proximity queries at different k
  - >= 1 query with a term absent from the corpus
  - written report of >= 2 cases where positional info changes the result set/order

None of the expected document IDs below are hard-coded as "correct answers" --
we simply RUN the queries against the live index and report what comes back,
per the assignment's "do not hard-code expected document IDs" instruction.

Novelty sections added on top of the required tests: rank-shift badges and
snippets are now part of every phrase/proximity row; a Custom SMART-notation
VSM comparison; and an NDCG@10/@5 report. NDCG needs *some* notion of
relevance and this corpus ships no qrels file, so see
_simulate_relevance_judgments() below for exactly how that gap is handled
without inventing "correct" doc IDs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.search_engine import SearchEngine
from src.preprocess import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(ROOT, "corpus_100.txt")
OUT_PATH = os.path.join(ROOT, "output", "test_results.md")

FREE_TEXT_QUERIES = [
    "cotton shirt for men",
    "winter jacket",
    "stretch denim jeans",
    "festive saree",
    "high waist leggings",
    "fleece hoodie black",
    "breathable fabric kurta",
    "regular fit sweatshirt",
    "printed dress women",
    "zip closure jacket",
]

PHRASE_QUERIES = [
    "cotton shirt",
    "stretch denim",
    "festive wear",
    "winter wear",
    "regular fit",
]

PROXIMITY_QUERIES = [
    ("cotton", "shirt", 3),
    ("stretch", "denim", 4),
    ("winter", "wear", 3),
    ("festive", "kurta", 4),
]

OOV_QUERY = "waterproof gore-tex trekking boots"

# Query set used for the NDCG novelty section (Part E extension).
NDCG_QUERIES = ["cotton shirt", "winter jacket", "high waist leggings"]

# tf.df.norm combinations demonstrated for the Custom SMART VSM novelty,
# always including the required lnc.ltc as the baseline/default.
CUSTOM_SCHEME_DEMOS = [("lnc", "ltc"), ("anc", "apc"), ("Lnu", "ltn"), ("bnn", "btn")]


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _simulate_relevance_judgments(engine, query, pool_docids):
    """
    NDCG needs relevance judgments, and this corpus has no supplied qrels
    file. Rather than hand-picking "correct" doc IDs (which Part E
    explicitly tells us not to do) or treating any one ranker's own
    output as ground truth (circular -- that would make NDCG measure
    nothing), we derive an explicit, disclosed, rule-based judge for
    this report only:
        3 = the query is an exact phrase match in the doc (positional index hit)
        2 = every distinct query term occurs somewhere in the doc (bag-of-words hit)
        1 = at least one query term occurs in the doc
        0 = no query term occurs in the doc
    This is a transparent, reproducible PROXY for demonstrating how NDCG
    differentiates the rankers -- it is not a claim that these are the
    "true" relevant documents. cli.py's NDCG menu lets a real human grade
    relevance instead, which is the honest way to use this metric outside
    of this illustrative report.
    """
    q_terms = set(preprocess(query))
    phrase_docids = {r["docid"] for r in engine.search_phrase(query, top_k=100)}
    judgments = {}
    for d in pool_docids:
        doc_terms = set(engine.corpus.doc_tokens[d])
        if d in phrase_docids:
            judgments[d] = 3
        elif q_terms and q_terms.issubset(doc_terms):
            judgments[d] = 2
        elif q_terms & doc_terms:
            judgments[d] = 1
        else:
            judgments[d] = 0
    return judgments


def main():
    engine = SearchEngine(CORPUS_PATH)
    lines = []
    lines.append("# Part E — Test Results\n")
    lines.append(f"Corpus: {engine.corpus.N} documents, "
                 f"{len(engine.inverted_index)} unique terms after stemming/stopword removal.\n")

    # ---------------- Free-text queries ----------------
    lines.append("## Free-text ranked queries (VSM lnc.ltc, top 10)\n")
    for q in FREE_TEXT_QUERIES:
        results, oov = engine.search_vsm(q, top_k=10)
        lines.append(f"### Query: \"{q}\"")
        if oov:
            lines.append(f"_OOV terms (not in corpus dictionary): {oov}_")
        if results:
            rows = [(r["docid"], r["category"], r["title"], r["score"]) for r in results]
            lines.append(fmt_table(rows, ["DocID", "Category", "Title", "Score"]))
        else:
            lines.append("_No matching documents._")
        lines.append("")

    # OOV query, run explicitly + typo-tolerance comparison
    lines.append(f"### Query containing an out-of-corpus term: \"{OOV_QUERY}\"")
    results, oov = engine.search_vsm(OOV_QUERY, top_k=10)
    lines.append(f"OOV terms: {oov}")
    if results:
        rows = [(r["docid"], r["category"], r["title"], r["score"]) for r in results]
        lines.append(fmt_table(rows, ["DocID", "Category", "Title", "Score"]))
    else:
        lines.append("_No matching documents (as expected — none of these query terms exist in this clothing corpus)._")
    tt_results, corrections = engine.search_with_typo_tolerance(OOV_QUERY, top_k=5)
    lines.append(f"\n_Novelty: typo-tolerant fallback suggested corrections: {corrections}_")
    lines.append("")

    # ---------------- Phrase queries (+ rank-shift, snippet novelties) ----------------
    lines.append("## Exact phrase queries (positional index)\n")
    lines.append("_Rank shift compares each result's position here to where plain VSM "
                 "lnc.ltc would rank the same query text (`^N` = moved up N places, "
                 "`vN` = moved down N places, `=` = unchanged, `NEW` = VSM didn't rank "
                 "this document at all)._\n")
    for p in PHRASE_QUERIES:
        results = engine.search_phrase(p, top_k=10)
        lines.append(f"### Phrase: \"{p}\"")
        if results:
            rows = [(r["docid"], r["category"], r["title"], r["match_positions"],
                      r["rank_shift"], r["snippet"]) for r in results]
            lines.append(fmt_table(rows, ["DocID", "Category", "Title", "Match positions",
                                            "Rank shift vs VSM", "Snippet"]))
        else:
            lines.append("_No exact phrase matches._")
        lines.append("")

    # ---------------- Proximity queries (+ rank-shift, snippet novelties) ----------------
    lines.append("## Proximity queries (ordered, WITHIN/k)\n")
    for a, b, k in PROXIMITY_QUERIES:
        results = engine.search_proximity(a, b, k, top_k=10)
        lines.append(f"### {a} WITHIN/{k} {b}")
        if results:
            rows = [(r["docid"], r["category"], r["title"], r["match_positions"],
                      r["rank_shift"], r["snippet"]) for r in results]
            lines.append(fmt_table(rows, ["DocID", "Category", "Title", "(pos_a,pos_b) pairs",
                                            "Rank shift vs VSM", "Snippet"]))
        else:
            lines.append("_No proximity matches within this window._")
        lines.append("")

    # ---------------- Where positional info changes results vs. plain VSM ----------------
    lines.append("## Where positional information changes results vs. plain VSM\n")
    lines.append(_case_study(engine, "regular fit"))
    lines.append(_zero_idf_case_study(engine, "festive wear"))

    lines.append("\n## Ranker comparison on one query (VSM vs BM25 vs SDM vs Fielded vs Custom)\n")
    q = "cotton shirt"
    for label, fn in [
        ("VSM lnc.ltc", lambda: engine.search_vsm(q, 5)[0]),
        ("BM25", lambda: engine.search_bm25(q, 5)[0]),
        ("SDM (VSM base)", lambda: engine.search_sdm(q, 5)[0]),
        ("Title/Category field-boosted VSM", lambda: engine.search_fielded(q, 5)[0]),
        ("Custom SMART VSM (lnc.ltc default)", lambda: engine.search_custom(q, 5)[0]),
    ]:
        res = fn()
        rows = [(r["docid"], r["category"], r["title"], r["score"]) for r in res]
        lines.append(f"### {label} — query \"{q}\"")
        lines.append(fmt_table(rows, ["DocID", "Category", "Title", "Score"]))
        lines.append("")

    # ---------------- Custom SMART-notation VSM scheme comparison (novelty) ----------------
    lines.append("## Custom SMART-notation VSM schemes (novelty)\n")
    lines.append("Demonstrates the generalized tf.df.norm builder (src/smart_vsm.py) against "
                 "the same query, including the required `lnc.ltc` as the baseline/default. "
                 "'u' (pivoted unique) and 'b' (byte size) norms are documented simplifications "
                 "-- see smart_vsm.py's `_normalize()` docstring.\n")
    demo_query = "cotton shirt"
    default_scheme = engine.custom_scheme
    for doc_s, q_s in CUSTOM_SCHEME_DEMOS:
        engine.set_custom_scheme(doc_s, q_s)
        results, oov = engine.search_custom(demo_query, top_k=5)
        lines.append(f"### Scheme {doc_s}.{q_s} — query \"{demo_query}\"")
        if results:
            rows = [(r["docid"], r["category"], r["title"], r["score"]) for r in results]
            lines.append(fmt_table(rows, ["DocID", "Category", "Title", "Score"]))
        else:
            lines.append("_No matching documents under this scheme._")
        lines.append("")
    if default_scheme:
        engine.set_custom_scheme(*default_scheme)  # restore default before NDCG section

    # ---------------- NDCG (novelty, Part E extension) ----------------
    lines.append("## NDCG@10 / NDCG@5 across rankers (novelty, Part E extension)\n")
    lines.append("_Relevance judgments below are a disclosed, rule-based proxy "
                 "(`_simulate_relevance_judgments()` in this file) -- this corpus ships with "
                 "no qrels file, so these are NOT hard-coded \"correct\" doc IDs, just a "
                 "transparent, reproducible stand-in for demonstrating how NDCG "
                 "differentiates the rankers. `cli.py`'s interactive NDCG menu lets a real "
                 "human grade relevance instead._\n")
    for q in NDCG_QUERIES:
        report = engine.ndcg_report(q, judgments={}, use_custom=True)
        pool = sorted({d for r in report.values() for d in r["ranked_docids"]})
        judgments = _simulate_relevance_judgments(engine, q, pool)
        report = engine.ndcg_report(q, judgments, use_custom=True)
        lines.append(f"### Query: \"{q}\" ({len(pool)} pooled documents)")
        rows = [(label, f"{r['ndcg_at_10']:.4f}", f"{r['ndcg_at_5']:.4f}")
                for label, r in sorted(report.items(), key=lambda kv: -kv[1]["ndcg_at_10"])]
        lines.append(fmt_table(rows, ["Ranker", "NDCG@10", "NDCG@5"]))
        lines.append("")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_PATH}")


def _case_study(engine, query):
    vsm_res, _ = engine.search_vsm(query, top_k=10)
    sdm_res, _, evidence = engine.search_sdm(query, top_k=10)
    vsm_order = [r["docid"] for r in vsm_res]
    sdm_order = [r["docid"] for r in sdm_res]

    out = [f"### Case study: \"{query}\"",
           f"- VSM (bag-of-words) top-10 order: {vsm_order}",
           f"- SDM (positional, ordered-bigram-aware) top-10 order: {sdm_order}"]

    if vsm_order == sdm_order:
        out.append("- Order unchanged for this query (every candidate document already has "
                    "the query terms adjacent, so there is no positional evidence to "
                    "differentiate them).")
        out.append("")
        return "\n".join(out)

    # Find the first rank position where the two orders diverge, and explain
    # WHY using the raw ordered-bigram evidence computed by SDM.
    divergence_rank = next(i for i in range(min(len(vsm_order), len(sdm_order)))
                            if vsm_order[i] != sdm_order[i])
    moved_up = [d for d in sdm_order[:divergence_rank + 3] if d not in vsm_order[:divergence_rank + 3]]
    moved_down = [d for d in vsm_order[:divergence_rank + 3] if d not in sdm_order[:divergence_rank + 3]]
    ordered_hits = evidence["ordered_hits"]

    out.append(f"- **Order changes starting at rank {divergence_rank + 1}.** Documents promoted "
               f"by SDM: {moved_up or 'none'} (ordered-bigram hit counts: "
               f"{ {d: ordered_hits.get(d, 0) for d in moved_up} }). Documents demoted: "
               f"{moved_down or 'none'} (ordered-bigram hit counts: "
               f"{ {d: ordered_hits.get(d, 0) for d in moved_down} }).")
    out.append("- Reason: bag-of-words VSM assigns the same score to any document containing "
               "both query terms regardless of where they sit in the text. SDM adds credit "
               "when the terms occur *adjacent and in query order* (using the positional "
               "index), so a document where they form an exact phrase outranks one where "
               "they are merely both present but scattered across the description. The "
               "rank-shift badges on the phrase-search table above show this same effect "
               "directly against plain VSM.")
    out.append("")
    return "\n".join(out)


def _zero_idf_case_study(engine, query):
    """
    A more dramatic case: both query terms occur in every single document
    of this templated corpus (df=100 => idf=0), so ltc/BM25-style
    weighting has ZERO discriminative power and returns no ranked
    matches at all -- yet the words form a meaningful literal phrase in
    a subset of documents, which the positional index (unaffected by
    idf) still finds. This is also exactly the case where the rank-shift
    badge shows "NEW" for every phrase-search result.
    """
    vsm_res, oov = engine.search_vsm(query, top_k=10)
    phrase_res = engine.search_phrase(query, top_k=10)
    sdm_res, _, _ = engine.search_sdm(query, top_k=10)

    out = [f"### Case study: \"{query}\" (result SET changes, not just order)",
           f"- VSM lnc.ltc result set: {[r['docid'] for r in vsm_res] or 'EMPTY -- see explanation below'}",
           f"- Exact phrase search result set: {[r['docid'] for r in phrase_res]}",
           f"- Rank-shift badges on those phrase results: "
           f"{[r['rank_shift'] for r in phrase_res]} (expect all \"NEW\" -- VSM never ranked them)",
           f"- SDM result set (uses positional evidence even when unigram idf=0): "
           f"{[r['docid'] for r in sdm_res]}"]
    out.append("- **Explanation:** every one of the 100 documents in this corpus contains "
               "both of these words somewhere in its templated description "
               "(df=100 for each term), so idf = log10(N/df) = 0 for both, which makes every "
               "ltc query weight zero and VSM/BM25-style scoring unable to distinguish ANY "
               "document from any other -- a real bag-of-words retrieval fails to notice that "
               "the phrase is actually a meaningful, literal attribute description in a "
               "specific subset of products. The positional index is unaffected by idf and "
               "still finds the documents where the words occur as a genuine adjacent phrase, "
               "so the result SET (not merely its order) changes from empty to populated.")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
