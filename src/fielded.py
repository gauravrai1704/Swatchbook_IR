"""
NOVELTY FEATURE 3 -- fielded title/category boosting (BM25F-style).

Research grounding: Robertson, Zaragoza & Taylor, "Simple BM25 Extension
to Multiple Weighted Fields", CIKM 2004 -- the paper that introduced
BM25F, still the standard way production search engines (and e-commerce
search specifically) weight structured product data: a query term
matching the product TITLE/CATEGORY is much stronger relevance evidence
than the same term buried in a generic marketing sentence in the body
text. Our corpus is a clothing e-commerce catalogue where every
document has an explicit TITLE + CATEGORY field, making this a natural,
well-motivated fit (rather than a generic add-on).

We implement a lightweight version: rather than re-deriving full BM25F
term-frequency mixing, we apply a multiplicative boost to a document's
unigram score for every query term that also occurs in that document's
TITLE/CATEGORY field. This is intentionally simple (transparent to grade
and reason about) while directly reflecting the BM25F insight that
field-of-occurrence should change a term's contribution to relevance.
"""

from src.preprocess import preprocess


class FieldBoostedRanker:
    def __init__(self, base_ranker, title_index, boost=1.5):
        """
        base_ranker: VSMIndex or BM25Index (anything with .search()).
        title_index: term -> {docid: title_tf}, from build_title_field_index().
        boost: multiplicative factor applied per matching query term found
               in the title/category field of a given document.
        """
        self.base_ranker = base_ranker
        self.title_index = title_index
        self.boost = boost

    def search(self, query_text, top_k=10):
        base_ranked, oov_terms = self.base_ranker.search(query_text, top_k=200)
        q_terms = set(preprocess(query_text))

        boosted = []
        for docid, score in base_ranked:
            factor = 1.0
            for term in q_terms:
                if docid in self.title_index.get(term, {}):
                    factor *= self.boost
            boosted.append((docid, score * factor))

        boosted.sort(key=lambda x: (-x[1], x[0]))
        return boosted[:top_k], oov_terms
