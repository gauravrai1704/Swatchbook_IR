"""
NOVELTY FEATURE 1 -- BM25 (Okapi) ranking.

lnc.ltc uses raw log-tf saturation but does NOT normalize for document
length beyond the cosine normalization of the whole vector, and it
gives idf-weighted credit only on the query side. BM25 (Robertson &
Walker, 1994; Robertson & Zaragoza, "The Probabilistic Relevance
Framework: BM25 and Beyond", Foundations & Trends in IR, 2009) is the
de facto industry-standard scoring function used by Lucene/Elasticsearch
and virtually every modern retrieval pipeline (including as the sparse
/lexical half of hybrid dense+sparse systems in current RAG research,
e.g. Sultania et al. 2024, "Contextual Text Embeddings" -- hybrid BM25 +
embeddings; Qiao et al. 2022 on learned-sparse + BM25 fusion). Adding a
BM25 ranker lets us empirically compare two term-weighting philosophies
on the same corpus/queries.

BM25 formula per term t in query, document d:
    score(d, t) = idf(t) * ( tf_td * (k1 + 1) ) / ( tf_td + k1 * (1 - b + b * |d| / avgdl) )
    idf(t) = log( (N - df_t + 0.5) / (df_t + 0.5) + 1 )   (Robertson-Sparck Jones, +1 smoothed variant)
"""

import math
from collections import Counter, defaultdict
from src.preprocess import preprocess


class BM25Index:
    def __init__(self, corpus, inverted_index, k1=1.5, b=0.75):
        self.corpus = corpus
        self.index = inverted_index
        self.N = corpus.N
        self.k1 = k1
        self.b = b
        self.doc_len = {docid: len(toks) for docid, toks in corpus.doc_tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))

    def idf(self, term):
        entry = self.index.get(term)
        df = entry["df"] if entry else 0
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1)

    def search(self, query_text, top_k=10):
        q_terms = preprocess(query_text)
        oov_terms = [t for t in set(q_terms) if t not in self.index]
        scores = defaultdict(float)
        q_tf = Counter(q_terms)

        for term in q_tf:
            entry = self.index.get(term)
            if not entry:
                continue
            idf = self.idf(term)
            for docid, tf in entry["postings"].items():
                dl = self.doc_len.get(docid, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[docid] += idf * (tf * (self.k1 + 1)) / (denom or 1)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k], oov_terms
