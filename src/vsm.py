"""
Part B: Ranked retrieval using the Vector Space Model, lnc.ltc scheme.

Document weight   (l,n,c -> "lnc"):  w_d,t = 1 + log10(tf_d,t)         [no idf, cosine-normalized]
Query weight       (l,t,c -> "ltc"): w_q,t = (1 + log10(tf_q,t)) * log10(N / df_t)   [cosine-normalized]

score(q, d) = sum_t  w_q,t * w_d,t     (vectors already length-normalized to unit length,
                                         so the dot product IS the cosine similarity)
"""

import math
from collections import Counter, defaultdict
from src.preprocess import preprocess


class VSMIndex:
    def __init__(self, corpus, inverted_index):
        self.corpus = corpus
        self.N = corpus.N
        self.index = inverted_index  # term -> {df, postings: {docid: tf}}

        # ---- Pre-compute normalized document vectors (the "lnc" side) ----
        # raw log-tf weights per doc
        raw_doc_weights = defaultdict(dict)  # docid -> {term: 1+log10(tf)}
        for term, entry in self.index.items():
            for docid, tf in entry["postings"].items():
                raw_doc_weights[docid][term] = 1.0 + math.log10(tf)

        # cosine length per doc, then normalize
        self.doc_vectors = {}  # docid -> {term: normalized weight}
        self.doc_norms = {}
        for docid, weights in raw_doc_weights.items():
            norm = math.sqrt(sum(w * w for w in weights.values()))
            self.doc_norms[docid] = norm
            if norm > 0:
                self.doc_vectors[docid] = {t: w / norm for t, w in weights.items()}
            else:
                self.doc_vectors[docid] = {}

        # make sure every doc (even one with zero indexed terms) has an entry
        for docid in self.corpus.doc_ids:
            self.doc_vectors.setdefault(docid, {})

    def idf(self, term):
        entry = self.index.get(term)
        if not entry or entry["df"] == 0:
            return 0.0
        return math.log10(self.N / entry["df"])

    def query_vector(self, query_text):
        """Builds the ltc-weighted, cosine-normalized query vector."""
        tokens = preprocess(query_text)
        tf_counts = Counter(tokens)
        raw = {}
        for term, tf in tf_counts.items():
            idf = self.idf(term)
            if idf <= 0:
                # term absent from corpus (df=0) or in every doc (idf=0) ->
                # contributes nothing to ltc weighting; still tracked so the
                # caller can report "no matching documents for term X".
                raw[term] = 0.0
            else:
                raw[term] = (1.0 + math.log10(tf)) * idf
        norm = math.sqrt(sum(w * w for w in raw.values())) or 1.0
        return {t: w / norm for t, w in raw.items()}, tokens

    def search(self, query_text, top_k=10):
        """
        Returns list of (docid, score) sorted by decreasing score, ties
        broken by increasing docID. Only computes over documents that
        contain at least one query term (matching documents), as required.
        """
        q_vec, q_tokens = self.query_vector(query_text)

        # Which query terms actually occur in the corpus (for reporting)
        oov_terms = [t for t in set(q_tokens) if t not in self.index]

        scores = defaultdict(float)
        for term, w_qt in q_vec.items():
            if w_qt == 0.0:
                continue
            entry = self.index.get(term)
            if not entry:
                continue
            for docid in entry["postings"]:
                w_dt = self.doc_vectors[docid].get(term, 0.0)
                scores[docid] += w_qt * w_dt

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k], oov_terms
