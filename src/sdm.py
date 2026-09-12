"""
NOVELTY FEATURE 2 -- Sequential Dependence Model (SDM), the centrepiece
novelty of this project.

Research grounding: Metzler, D. and Croft, W.B., "A Markov Random Field
Model for Term Dependencies", SIGIR 2005 (ACM), one of the most-cited
papers in ad-hoc IR, later implemented as the #sdm/#seqdep operator in
the Lemur/Indri/Galago retrieval toolkits (Univ. of Massachusetts CIIR).
SDM is a special case of a general Markov Random Field over query terms
and treats a query as evidence from three feature classes:
    f_T  -- single term (unigram) match       -> our VSM/BM25 score
    f_O  -- ORDERED bigram ("exact phrase")    -> our positional index, phrase_search
    f_U  -- UNORDERED window (bag-of-2, near)  -> our positional index, proximity_search
and linearly combines them:
    SDM(Q, D) = lambda_T * sum_unigrams + lambda_O * sum_ordered_bigrams
                                        + lambda_U * sum_unordered_bigrams
Lemur's default weights (lambda_T=0.8, lambda_O=0.15, lambda_U=0.05) are
used here as a reasonable, published starting point (see Lemur Project
wiki, "Galago Operators": SequentialDependenceTraversal).

Why this matters for THIS assignment specifically: the assignment asks
us to "clearly understand the difference between ordinary VSM retrieval
and positional retrieval" (Part C) but only uses positions for
Boolean-style phrase/proximity filtering. SDM goes one step further and
folds positional evidence directly INTO the ranking score, so a document
where "cotton" and "shirt" appear adjacent (e.g. "cotton shirt") is
ranked above one where they appear far apart, even though a pure
bag-of-words VSM/BM25 score would treat both documents identically
(both simply "contain cotton" and "contain shirt" once each).
"""

import math
from collections import defaultdict
from src.preprocess import preprocess
from src.positional_search import candidate_docs


def _ordered_bigram_count(positional_index, t1, t2, docid, window=2):
    """
    Counts (t1, t2) occurrences with 0 < gap <= window and t1 before t2,
    within a single document -- this is f_O for one bigram in one doc.
    """
    p1 = positional_index[t1]["postings"][docid]["positions"]
    p2 = positional_index[t2]["postings"][docid]["positions"]
    count = 0
    for a in p1:
        for b in p2:
            if 0 < (b - a) <= window:
                count += 1
    return count


def _unordered_window_count(positional_index, t1, t2, docid, window=4):
    """f_U for one bigram in one doc: unordered proximity within `window`."""
    p1 = positional_index[t1]["postings"][docid]["positions"]
    p2 = positional_index[t2]["postings"][docid]["positions"]
    count = 0
    for a in p1:
        for b in p2:
            if 0 < abs(b - a) <= window:
                count += 1
    return count


class SDMRanker:
    def __init__(self, corpus, positional_index, unigram_ranker,
                 lambda_t=0.8, lambda_o=0.15, lambda_u=0.05,
                 ordered_window=2, unordered_window=4):
        """
        unigram_ranker: any object exposing .search(query_text, top_k) ->
        (ranked_list, oov_terms) -- we reuse VSMIndex or BM25Index for
        the f_T term, so unigram weighting philosophy is pluggable.
        """
        self.corpus = corpus
        self.pindex = positional_index
        self.unigram_ranker = unigram_ranker
        self.lambda_t = lambda_t
        self.lambda_o = lambda_o
        self.lambda_u = lambda_u
        self.ordered_window = ordered_window
        self.unordered_window = unordered_window

    def search(self, query_text, top_k=10, candidate_pool=50):
        q_terms = preprocess(query_text)
        oov_terms = [t for t in set(q_terms) if t not in self.pindex]

        # 1. unigram evidence: take a generous candidate pool from the
        #    plugged-in unigram ranker (VSM or BM25), normalized to [0,1]
        #    by dividing by the max score so it's comparable across queries.
        unigram_ranked, _ = self.unigram_ranker.search(query_text, top_k=candidate_pool)
        max_u = max((s for _, s in unigram_ranked), default=0.0) or 1.0
        unigram_scores = {docid: s / max_u for docid, s in unigram_ranked}

        # any doc containing >=2 distinct query terms is also a candidate
        # for bigram evidence even if it wasn't in the top unigram pool
        distinct_terms = list(dict.fromkeys(q_terms))
        bigram_candidate_docs = set(unigram_scores.keys())
        for i in range(len(distinct_terms) - 1):
            t1, t2 = distinct_terms[i], distinct_terms[i + 1]
            if t1 in self.pindex and t2 in self.pindex:
                bigram_candidate_docs |= candidate_docs(self.pindex, [t1, t2])

        ordered_scores = defaultdict(int)
        unordered_scores = defaultdict(int)
        n_bigrams = max(1, len(distinct_terms) - 1)

        for i in range(len(distinct_terms) - 1):
            t1, t2 = distinct_terms[i], distinct_terms[i + 1]
            if t1 not in self.pindex or t2 not in self.pindex:
                continue
            docs = candidate_docs(self.pindex, [t1, t2])
            for docid in docs:
                ordered_scores[docid] += _ordered_bigram_count(
                    self.pindex, t1, t2, docid, self.ordered_window)
                unordered_scores[docid] += _unordered_window_count(
                    self.pindex, t1, t2, docid, self.unordered_window)

        max_o = max(ordered_scores.values(), default=0) or 1
        max_un = max(unordered_scores.values(), default=0) or 1

        final = {}
        all_docs = set(unigram_scores) | set(ordered_scores) | set(unordered_scores)
        for docid in all_docs:
            ft = unigram_scores.get(docid, 0.0)
            fo = ordered_scores.get(docid, 0) / max_o
            fu = unordered_scores.get(docid, 0) / max_un
            final[docid] = (self.lambda_t * ft +
                             self.lambda_o * fo +
                             self.lambda_u * fu)

        ranked = sorted(final.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k], oov_terms, {
            "ordered_hits": dict(ordered_scores),
            "unordered_hits": dict(unordered_scores),
        }
