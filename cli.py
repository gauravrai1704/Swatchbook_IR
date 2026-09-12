"""
Part D: Application interface.

Modes:
  1) Free-text ranked retrieval (VSM lnc.ltc, plus BM25/SDM/fielded/
     typo-tolerant/custom-SMART rankers for comparison, each result
     shown with a highlighted snippet)
  2) Phrase / proximity search over the positional index -- now also
     shows a rank-shift badge (^N / vN / NEW) comparing each result's
     position to where plain VSM would have ranked the same query text
  3) Relevance & NDCG -- grade your own relevance judgments for a query
     and see NDCG@10 / NDCG@5 compared across every ranker
  4) Custom SMART-notation VSM builder -- pick tf/df/norm codes for the
     document side and the query side independently and search with it

Run:
    python3 cli.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.search_engine import SearchEngine
from src.smart_vsm import TF_LABELS, DF_LABELS, NORM_LABELS

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_100.txt")


def print_ranked(results, oov=None):
    if oov:
        print(f"  (note: term(s) not found in corpus dictionary: {', '.join(oov)})")
    if not results:
        print("  No matching documents.")
        return
    print(f"  {'Rank':<5}{'DocID':<8}{'Score':<9}{'Category':<14}Title")
    for i, r in enumerate(results, 1):
        print(f"  {i:<5}{r['docid']:<8}{r['score']:<9}{r['category']:<14}{r['title']}")
        if r.get("snippet"):
            print(f"           {r['snippet']}")


def print_positional(results):
    if not results:
        print("  No matching documents.")
        return
    for r in results:
        badge = f"  [{r['rank_shift']} vs. plain VSM]" if r.get("rank_shift") else ""
        print(f"  {r['docid']:<8}{r['category']:<14}{r['title']}{badge}")
        print(f"           match positions: {r['match_positions']}")
        if r.get("snippet"):
            print(f"           {r['snippet']}")


def free_text_menu(engine):
    query = input("Enter free-text query: ").strip()
    if not query:
        return
    print("\n[VSM lnc.ltc]  (required Part B ranking)")
    results, oov = engine.search_vsm(query)
    print_ranked(results, oov)

    print("\n[Novelty] BM25")
    results, oov = engine.search_bm25(query)
    print_ranked(results, oov)

    print("\n[Novelty] Sequential Dependence Model (unigram+phrase+proximity, on VSM base)")
    results, oov, evidence = engine.search_sdm(query, base="vsm")
    print_ranked(results, oov)

    print("\n[Novelty] Title/Category field-boosted VSM")
    results, oov = engine.search_fielded(query)
    print_ranked(results, oov)

    scheme = engine.custom_scheme or ("lnc", "ltc")
    print(f"\n[Novelty] Custom SMART VSM ({scheme[0]}.{scheme[1]}) -- change via menu option 4")
    results, oov = engine.search_custom(query)
    print_ranked(results, oov)

    if oov:
        print("\n[Novelty] Typo-tolerant retry (edit-distance correction)")
        results, corrections = engine.search_with_typo_tolerance(query)
        if corrections:
            print(f"  corrected: {corrections}")
        print_ranked(results)


def positional_menu(engine):
    print("1) Exact phrase search   2) Proximity search (WITHIN/k)")
    choice = input("Choose mode: ").strip()
    if choice == "1":
        phrase = input("Enter phrase (e.g. 'cotton shirt'): ").strip()
        print(f"\n[Positional] Exact phrase search: \"{phrase}\"")
        print_positional(engine.search_phrase(phrase))
    else:
        term_a = input("First term: ").strip()
        term_b = input("Second term: ").strip()
        k = int(input("Within how many positions (k): ").strip() or "3")
        print(f"\n[Positional] Proximity search: {term_a} WITHIN/{k} {term_b}")
        print_positional(engine.search_proximity(term_a, term_b, k))


def ndcg_menu(engine):
    """NOVELTY: interactive relevance grading + live NDCG comparison.
    Judgments come entirely from whoever is running the CLI right now --
    nothing here is pre-filled or hard-coded."""
    query = input("Enter free-text query to judge: ").strip()
    if not query:
        return
    report = engine.ndcg_report(query, judgments={}, use_custom=True)
    pool = sorted({d for r in report.values() for d in r["ranked_docids"]})
    if not pool:
        print("  No matching documents across any ranker for this query.")
        return

    print(f"\nPooled top-10 documents across {len(report)} rankers ({len(pool)} unique). "
          "Grade each 0 (not relevant) to 3 (highly relevant); blank = 0.")
    judgments = {}
    for d in pool:
        title = engine.corpus.doc_by_id[d]["title"]
        grade = input(f"  {d} - {title}\n    relevance (0-3): ").strip()
        judgments[d] = int(grade) if grade.isdigit() and 0 <= int(grade) <= 3 else 0

    report = engine.ndcg_report(query, judgments, use_custom=True)
    print(f"\n[NDCG] \"{query}\" -- your own relevance grades, nothing hard-coded")
    print(f"  {'Ranker':<24}{'NDCG@10':<10}NDCG@5")
    for label, r in sorted(report.items(), key=lambda kv: -kv[1]["ndcg_at_10"]):
        print(f"  {label:<24}{r['ndcg_at_10']:<10.4f}{r['ndcg_at_5']:.4f}")


def custom_vsm_menu(engine):
    """NOVELTY: build any tf.df.norm SMART scheme for document and query
    sides independently; lnc.ltc (the required Part B scheme) is the
    starting default and is always reachable by pressing Enter twice."""
    scheme = engine.custom_scheme or ("lnc", "ltc")
    print(f"Current scheme: {scheme[0]}.{scheme[1]}")
    print("tf codes:   " + ", ".join(f"{k}={v}" for k, v in TF_LABELS.items()))
    print("df codes:   " + ", ".join(f"{k}={v}" for k, v in DF_LABELS.items()))
    print("norm codes: " + ", ".join(f"{k}={v}" for k, v in NORM_LABELS.items()))
    doc_scheme = input("Document scheme (3 chars, e.g. lnc) [lnc]: ").strip() or "lnc"
    query_scheme = input("Query scheme (3 chars, e.g. ltc) [ltc]: ").strip() or "ltc"
    try:
        engine.set_custom_scheme(doc_scheme, query_scheme)
    except KeyError as e:
        print(f"  Unknown code {e}; scheme left unchanged ({scheme[0]}.{scheme[1]}).")
        return
    print(f"  Custom scheme set to {doc_scheme}.{query_scheme}. "
          "Use option 1 (free-text search) to see it ranked.")


def main():
    print("Loading clothing search engine (100-document corpus)...")
    engine = SearchEngine(CORPUS_PATH)
    print(f"Indexed {engine.corpus.N} documents, {len(engine.inverted_index)} unique terms.\n")

    while True:
        print("=" * 60)
        print("1) Free-text ranked search   2) Phrase/proximity search")
        print("3) Relevance & NDCG          4) Custom SMART VSM builder   5) Quit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            free_text_menu(engine)
        elif choice == "2":
            positional_menu(engine)
        elif choice == "3":
            ndcg_menu(engine)
        elif choice == "4":
            custom_vsm_menu(engine)
        else:
            break


if __name__ == "__main__":
    main()
