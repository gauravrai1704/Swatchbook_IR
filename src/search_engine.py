"""
Facade tying together Parts A-D plus the novelty rankers, so the CLI
(Part D) and the test harness (Part E) both go through one clean API.

Backend port note: this facade now mirrors swatchbook.html's
SearchEngine JS class feature-for-feature -- custom SMART-notation VSM
(src/smart_vsm.py), highlighted snippets (src/snippets.py), rank-shift
badges vs. plain VSM (src/rank_shift.py), and NDCG reporting
(src/ndcg.py) -- so the CLI and the web demo stay in sync.
"""

from src.index_builder import (
    Corpus, build_positional_index, build_inverted_index_from_positional,
    build_title_field_index,
)
from src.vsm import VSMIndex
from src.bm25 import BM25Index
from src.sdm import SDMRanker
from src.fielded import FieldBoostedRanker
from src.smart_vsm import SmartVSMIndex
from src.positional_search import phrase_search, proximity_search
from src.fuzzy import correct_query
from src.snippets import build_snippet
from src.rank_shift import compute_baseline_rank_map, rank_shift_badge
from src.ndcg import ndcg_at_k
from src.preprocess import preprocess


class SearchEngine:
    def __init__(self, corpus_path):
        # Build the core indexes once for the whole search engine lifecycle.
        # The positional index is needed for phrase/proximity queries, while the
        # inverted index supports the term-based vector-space and BM25 models.
        self.corpus = Corpus(corpus_path)
        self.positional_index = build_positional_index(self.corpus)
        self.inverted_index = build_inverted_index_from_positional(self.positional_index)
        self.title_index = build_title_field_index(self.corpus)

        # Rankers that operate over the full corpus and a shared vocabulary.
        self.vsm = VSMIndex(self.corpus, self.inverted_index)
        self.bm25 = BM25Index(self.corpus, self.inverted_index)
        self.sdm_on_vsm = SDMRanker(self.corpus, self.positional_index, self.vsm)
        self.sdm_on_bm25 = SDMRanker(self.corpus, self.positional_index, self.bm25)
        self.fielded_vsm = FieldBoostedRanker(self.vsm, self.title_index)

        # NOVELTY: custom SMART-notation VSM. Defaults to lnc.ltc (identical
        # to Part B's required scheme, verified in run_tests.py) so it's
        # always available even before a caller customizes it.
        self.custom_vsm = None
        self.custom_scheme = None
        self.set_custom_scheme("lnc", "ltc")

    def set_custom_scheme(self, doc_scheme, query_scheme):
        # Allow callers to swap in a different SMART weighting scheme without
        # rebuilding the corpus or the inverted index.
        self.custom_scheme = (doc_scheme, query_scheme)
        self.custom_vsm = SmartVSMIndex(self.corpus, self.inverted_index, doc_scheme, query_scheme)

    # ---------------- Part B: free-text ranked retrieval ----------------
    def search_vsm(self, query, top_k=10):
        # Standard vector-space retrieval using the core inverted index.
        ranked, oov = self.vsm.search(query, top_k)
        return self._format(ranked, query), oov

    def search_bm25(self, query, top_k=10):
        # BM25 is a common probabilistic baseline and is useful for comparison.
        ranked, oov = self.bm25.search(query, top_k)
        return self._format(ranked, query), oov

    def search_sdm(self, query, top_k=10, base="vsm"):
        # Sequential dependence model captures term proximity in the ranking.
        ranker = self.sdm_on_vsm if base == "vsm" else self.sdm_on_bm25
        ranked, oov, evidence = ranker.search(query, top_k)
        return self._format(ranked, query), oov, evidence

    def search_fielded(self, query, top_k=10):
        # Title-aware ranking boosts documents whose title matches the query.
        ranked, oov = self.fielded_vsm.search(query, top_k)
        return self._format(ranked, query), oov

    def search_custom(self, query, top_k=10):
        """NOVELTY: ranked retrieval under whatever SMART scheme was last
        set via set_custom_scheme() (defaults to lnc.ltc)."""
        # The custom SMART scheme remains available for experiments without
        # needing to recreate the engine object.
        if self.custom_vsm is None:
            self.set_custom_scheme("lnc", "ltc")
        ranked, oov = self.custom_vsm.search(query, top_k)
        return self._format(ranked, query), oov

    def search_with_typo_tolerance(self, query, top_k=10):
        # Correct likely spelling errors before ranking, while returning the
        # suggested corrections to the caller.
        corrected_tokens, corrections = correct_query(query, self.inverted_index)
        corrected_query = " ".join(corrected_tokens)
        ranked, oov = self.vsm.search(corrected_query, top_k)
        return self._format(ranked, corrected_query), corrections

    # ---------------- Part C: positional retrieval (+ rank-shift novelty) ----------------
    def search_phrase(self, phrase, top_k=10):
        # Phrase queries rely on exact positional matches; we sort by the number
        # of occurrences and then compare against the VSM baseline for novelty.
        matches = phrase_search(self.positional_index, phrase)
        ranked = sorted(matches.items(), key=lambda x: (-len(x[1]), x[0]))[:top_k]
        baseline_map = compute_baseline_rank_map(self.vsm, phrase)
        q_terms = preprocess(phrase)
        out = []
        for i, (d, pos) in enumerate(ranked):
            out.append({
                "docid": d, "title": self.corpus.doc_by_id[d]["title"],
                "category": self.corpus.doc_by_id[d]["category"],
                "match_positions": pos,
                "rank_shift": rank_shift_badge(d, i + 1, baseline_map),
                "vsm_rank": baseline_map.get(d),
                "snippet": build_snippet(self.corpus, d, q_terms),
            })
        return out

    def search_proximity(self, term_a, term_b, k, ordered=True, top_k=10):
        # Proximity search looks for near co-occurrence of two terms, still
        # returning a ranked list and a VSM comparison badge for novelty.
        matches = proximity_search(self.positional_index, term_a, term_b, k, ordered)
        ranked = sorted(matches.items(), key=lambda x: (-len(x[1]), x[0]))[:top_k]
        baseline_map = compute_baseline_rank_map(self.vsm, f"{term_a} {term_b}")
        q_terms = preprocess(f"{term_a} {term_b}")
        out = []
        for i, (d, pairs) in enumerate(ranked):
            out.append({
                "docid": d, "title": self.corpus.doc_by_id[d]["title"],
                "category": self.corpus.doc_by_id[d]["category"],
                "match_positions": pairs,
                "rank_shift": rank_shift_badge(d, i + 1, baseline_map),
                "vsm_rank": baseline_map.get(d),
                "snippet": build_snippet(self.corpus, d, q_terms),
            })
        return out

    # ---------------- NDCG (Part E extension, novelty) ----------------
    def ndcg_report(self, query, judgments, top_k=10, use_custom=False):
        """
        judgments: dict docid -> relevance grade (0-3), supplied entirely
        by the caller -- this method never invents or hard-codes one
        (see src/ndcg.py's module docstring).
        Returns dict ranker_label -> {"ndcg_at_10", "ndcg_at_5", "ranked_docids"}.
        """
        # Compare multiple retrieval strategies using the same query and grading
        # set, then summarize each ranker's quality with NDCG.
        rankers = [
            ("VSM (lnc.ltc)", lambda q, k: [d for d, _ in self.vsm.search(q, k)[0]]),
            ("BM25", lambda q, k: [d for d, _ in self.bm25.search(q, k)[0]]),
            ("SDM", lambda q, k: [d for d, _ in self.sdm_on_vsm.search(q, k)[0]]),
            ("Title-boosted VSM", lambda q, k: [d for d, _ in self.fielded_vsm.search(q, k)[0]]),
        ]
        if use_custom and self.custom_vsm is not None:
            label = f"Custom ({self.custom_scheme[0]}.{self.custom_scheme[1]})"
            rankers.append((label, lambda q, k: [d for d, _ in self.custom_vsm.search(q, k)[0]]))

        report = {}
        for label, fn in rankers:
            docids = fn(query, top_k)
            report[label] = {
                "ndcg_at_10": ndcg_at_k(docids, judgments, 10),
                "ndcg_at_5": ndcg_at_k(docids, judgments, 5),
                "ranked_docids": docids,
            }
        return report

    # ---------------- helpers ----------------
    def _format(self, ranked, query=None):
        # Standardize the raw ranking output so callers receive a consistent,
        # user-friendly document structure with optional snippets.
        q_terms = preprocess(query) if query else []
        out = []
        for docid, score in ranked:
            d = self.corpus.doc_by_id[docid]
            entry = {"docid": docid, "title": d["title"],
                     "category": d["category"], "score": round(score, 4)}
            if query:
                entry["snippet"] = build_snippet(self.corpus, docid, q_terms)
            out.append(entry)
        return out
