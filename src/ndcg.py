"""
NOVELTY -- NDCG (Jarvelin & Kekalainen, "Cumulated Gain-Based Evaluation
of IR Techniques", ACM TOIS 2002). Backend port of swatchbook.html's
dcgAtK/ndcgAtK, so CLI and web numbers agree exactly for the same
(ranking, judgments) input.

Honesty note: this corpus ships with no relevance-judgment (qrels)
file, and Part E explicitly says not to hard-code "correct" doc IDs.
NDCG has no meaning without *some* notion of relevance, so every caller
of ndcg_at_k() here must supply its own `judgments` dict explicitly --
this module never invents one. See run_tests.py's
_simulate_relevance_judgments() for a disclosed, rule-based proxy used
only to demonstrate the metric, and cli.py's NDCG menu for a real
human-graded alternative.
"""

import math


def dcg_at_k(relevances, k):
    """DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1), i 1-indexed."""
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_docids, judgments, k):
    """
    ranked_docids: ordered list of docIDs as returned by a ranker (best first).
    judgments: dict docid -> integer relevance grade (0-3, standard graded
               relevance). Docs missing from `judgments` default to 0
               (the IR-standard "unjudged = not relevant" convention).
    """
    relevances = [judgments.get(d, 0) for d in ranked_docids]
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0
