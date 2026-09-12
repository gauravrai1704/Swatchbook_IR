"""
NOVELTY -- backend port of swatchbook.html's Custom SMART-notation VSM
builder. Part B requires exactly lnc.ltc (log-tf, no-idf, cosine on the
document side; log-tf, idf, cosine on the query side); this generalizes
that into the full SMART notation (Salton & Buckley, "Term-weighting
approaches in automatic text retrieval", Information Processing &
Management, 1988) so any tf.df.norm combination can be composed for the
document side and query side independently.

src/vsm.py is untouched and remains the canonical, required Part B
implementation -- this module is an additive novelty built on the same
inverted index, not a replacement.
"""

import math
from collections import Counter, defaultdict
from src.preprocess import preprocess


# ---------------- Term-frequency components ----------------
def _tf_natural(tf, maxtf=0, avgtf=0):
    return tf


def _tf_log(tf, maxtf=0, avgtf=0):
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


def _tf_augmented(tf, maxtf=0, avgtf=0):
    return 0.5 + 0.5 * tf / maxtf if maxtf > 0 else 0.0


def _tf_boolean(tf, maxtf=0, avgtf=0):
    return 1.0 if tf > 0 else 0.0


def _tf_log_average(tf, maxtf=0, avgtf=0):
    if tf > 0 and avgtf > 0:
        return (1.0 + math.log10(tf)) / (1.0 + math.log10(avgtf))
    return 0.0


TF_FUNCS = {"n": _tf_natural, "l": _tf_log, "a": _tf_augmented, "b": _tf_boolean, "L": _tf_log_average}
TF_LABELS = {"n": "natural", "l": "logarithm", "a": "augmented", "b": "boolean", "L": "log average"}


# ---------------- Document-frequency components ----------------
def _df_none(df, N):
    return 1.0


def _df_idf(df, N):
    return math.log10(N / df) if df > 0 else 0.0


def _df_prob_idf(df, N):
    return max(0.0, math.log10((N - df) / df)) if df > 0 else 0.0


DF_FUNCS = {"n": _df_none, "t": _df_idf, "p": _df_prob_idf}
DF_LABELS = {"n": "no idf", "t": "idf", "p": "prob idf"}

NORM_LABELS = {"n": "none", "c": "cosine", "u": "pivoted unique*", "b": "byte size"}


class SmartVSMIndex:
    """
    doc_scheme / query_scheme: 3-character SMART codes, e.g. "lnc", "ltc".
    Position 0 = term-frequency component, 1 = document-frequency
    component, 2 = normalization. lnc.ltc (the default) reproduces
    src/vsm.py's scores exactly -- see run_tests.py for a check of that.
    """

    def __init__(self, corpus, inverted_index, doc_scheme="lnc", query_scheme="ltc"):
        self.corpus = corpus
        self.index = inverted_index
        self.N = corpus.N
        self.doc_scheme = doc_scheme
        self.query_scheme = query_scheme

        # Per-document stats needed by the richer tf variants (a, L) and
        # norms (u, b).
        self.max_tf, self.avg_tf, self.unique_terms, self.char_len = {}, {}, {}, {}
        for docid, tokens in corpus.doc_tokens.items():
            counts = Counter(tokens)
            tfs = list(counts.values())
            self.max_tf[docid] = max(tfs) if tfs else 0
            self.avg_tf[docid] = (sum(tfs) / len(tfs)) if tfs else 0.0
            self.unique_terms[docid] = len(tfs)
            d = corpus.doc_by_id[docid]
            self.char_len[docid] = len(d["title"]) + 1 + len(d["text"])

        self.doc_vectors = self._build_doc_vectors()

    def _raw_weight(self, tf, df, docid, scheme):
        tf_w = TF_FUNCS[scheme[0]](tf, self.max_tf[docid], self.avg_tf[docid])
        df_w = DF_FUNCS[scheme[1]](df, self.N)
        return tf_w * df_w

    def _normalize(self, weights, norm_code, docid=None, fallback_len=1):
        """
        NOTE (honesty flag, matches the corresponding JS comment in
        swatchbook.html): 'u' (pivoted unique) is simplified here to
        1 / (unique term count in the document/query). True pivoted
        normalization (Singhal, Buckley & Mitra, SIGIR 1996) tunes a
        slope against the corpus-average unique-term count; that
        calibration is NOT implemented, so 'u' is directionally right
        (penalizes documents/queries with many distinct terms) but not
        the textbook formula. 'b' (byte size) uses char length ^ 0.75,
        per the slide's CharLength^alpha, alpha < 1.
        """
        if norm_code == "n" or not weights:
            return dict(weights)
        if norm_code == "c":
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            return {t: w / norm for t, w in weights.items()}
        if norm_code == "u":
            u = (self.unique_terms.get(docid, 0) if docid is not None else fallback_len) or 1
            return {t: w / u for t, w in weights.items()}
        if norm_code == "b":
            length = (self.char_len.get(docid, 0) if docid is not None else fallback_len) or 1
            return {t: w / (length ** 0.75) for t, w in weights.items()}
        return dict(weights)

    def _build_doc_vectors(self):
        raw = defaultdict(dict)
        for term, entry in self.index.items():
            for docid, tf in entry["postings"].items():
                raw[docid][term] = self._raw_weight(tf, entry["df"], docid, self.doc_scheme)
        vectors = {}
        for docid in self.corpus.doc_ids:
            vectors[docid] = self._normalize(raw.get(docid, {}), self.doc_scheme[2], docid=docid)
        return vectors

    def query_vector(self, query_text):
        tokens = preprocess(query_text)
        tf_counts = Counter(tokens)
        tf_vals = list(tf_counts.values())
        maxq = max(tf_vals) if tf_vals else 0
        avgq = (sum(tf_vals) / len(tf_vals)) if tf_vals else 0.0

        raw = {}
        for term, tf in tf_counts.items():
            entry = self.index.get(term)
            df = entry["df"] if entry else 0
            tf_w = TF_FUNCS[self.query_scheme[0]](tf, maxq, avgq)
            df_w = DF_FUNCS[self.query_scheme[1]](df, self.N)
            raw[term] = tf_w * df_w

        fallback_len = len(query_text) or len(raw) or 1
        vec = self._normalize(raw, self.query_scheme[2], docid=None, fallback_len=fallback_len)
        oov_terms = [t for t in set(tokens) if t not in self.index]
        return vec, tokens, oov_terms

    def search(self, query_text, top_k=10):
        q_vec, _, oov_terms = self.query_vector(query_text)
        scores = defaultdict(float)
        for term, w_qt in q_vec.items():
            if not w_qt:
                continue
            entry = self.index.get(term)
            if not entry:
                continue
            for docid in entry["postings"]:
                w_dt = self.doc_vectors.get(docid, {}).get(term, 0.0)
                if w_dt:
                    scores[docid] += w_qt * w_dt
        ranked = sorted(
            ((d, s) for d, s in scores.items() if s != 0),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:top_k], oov_terms
