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
    """BM25 ranking over a prebuilt inverted index.

    This class is responsible for ranking documents by how strongly a query term
    matches a document, while accounting for:
      - how rare the term is across the corpus (IDF)
      - how often it appears in a particular document (term frequency)
      - how long the document is compared with the average document length

    In other words, we reward terms that are informative and frequent in a short,
    relevant document, while down-weighting very common terms and long documents.
    """

    def __init__(self, corpus, inverted_index, k1=1.5, b=0.75):
        # Keep references to the corpus and the term dictionary we already built.
        self.corpus = corpus
        self.index = inverted_index

        # Number of documents in the collection and BM25 tuning parameters.
        self.N = corpus.N
        self.k1 = k1
        self.b = b

        # Document length is the number of tokenized terms in each document.
        # This lets us normalize by document length in the BM25 denominator.
        self.doc_len = {docid: len(toks) for docid, toks in corpus.doc_tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(1, len(self.doc_len))

    def idf(self, term):
        """Return the inverse document frequency for a single term.

        A term that appears in very few documents is considered more informative,
        so it gets a larger IDF weight. The +0.5 smoothing keeps the formula
        stable even for rare or unseen words.
        """
        entry = self.index.get(term)
        df = entry["df"] if entry else 0
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1)

    def search(self, query_text, top_k=10):
        """Return the top documents for a query with BM25 scores.

        The ranking pipeline is:
        1. tokenize/normalize the user query
        2. collect query-term counts and identify OOV terms not in the index
        3. for each query term, add BM25 contribution to each matching document
        4. sort the accumulated scores and return the strongest matches
        """
        # Normalize query text into the same token shape used during indexing.
        q_terms = preprocess(query_text)

        # Terms absent from the inverted index cannot contribute to BM25 scoring.
        # We surface them so the caller can decide how to handle them later.
        oov_terms = [t for t in set(q_terms) if t not in self.index]

        # Aggregate repeated query terms so we do not double-count them in the same way
        # a raw bag-of-words representation would.
        scores = defaultdict(float)
        q_tf = Counter(q_terms)

        for term in q_tf:
            entry = self.index.get(term)
            if not entry:
                continue

            # IDF tells us how discriminative the term is in the full corpus.
            idf = self.idf(term)

            # Each posting contains a document ID and the term frequency in that doc.
            # We add the BM25 contribution from this term to the running document score.
            for docid, tf in entry["postings"].items():
                dl = self.doc_len.get(docid, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[docid] += idf * (tf * (self.k1 + 1)) / (denom or 1)

        # Sort by descending score, then by document ID as a deterministic tie-breaker.
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k], oov_terms
