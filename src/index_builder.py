"""
Part A: inverted index (term -> df -> postings[(docID, tf)])
Part C: positional index (term -> df -> postings[(docID, tf, [positions])])

We build the positional index once and DERIVE the plain inverted index
from it (a positional index is a strict superset of a term-frequency
index), so the two are always consistent with each other -- this also
mirrors how real search engines are built (Manning/Raghavan/Schuetze,
IIR ch.2).

Novelty add-on: a lightweight FIELDED index that separately records,
for each (docID, term), how many times the term occurred in the TITLE
field. This underlies the fielded/title-boost novelty feature in
src/fielded.py (see that file for the research grounding).
"""

from collections import defaultdict
from src.preprocess import preprocess, parse_corpus


class Corpus:
    """Container for the raw document collection plus the tokenized forms we need for search.

    The corpus is the central data model: every downstream index should be built from
    this same set of documents. That gives us a single, consistent representation of
    the collection and makes it easy to derive multiple index types without drift.
    """

    def __init__(self, path):
        # Raw documents parsed from disk. Each record contains fields like docid, title,
        # category, and text, which later get normalized into search-friendly tokens.
        self.docs = parse_corpus(path)
        self.doc_ids = [d["docid"] for d in self.docs]
        self.N = len(self.docs)
        self.doc_by_id = {d["docid"]: d for d in self.docs}

        # Tokenized fields, per document, after the full preprocessing pipeline.
        # These are the exact token streams used by retrieval and ranking code.
        self.doc_tokens = {}        # docid -> [stemmed tokens] (title + text)
        self.title_tokens = {}      # docid -> [stemmed tokens] (title only)

        for d in self.docs:
            # We include both title and body text in the main document token stream so
            # retrieval considers the full document content, not just the title.
            full_text = d["title"] + ". " + d["text"]
            self.doc_tokens[d["docid"]] = preprocess(full_text)

            # For title-aware ranking, we keep a separate token stream built from the
            # title and category fields only; this lets us weight title terms differently
            # when implementing fielded relevance features.
            self.title_tokens[d["docid"]] = preprocess(d["title"] + " " + d["category"])


def build_positional_index(corpus: Corpus):
    """Build the richest index format first, then derive simpler ones from it.

    A positional index stores, for each term, the documents it occurs in and the exact
    token positions within each document. This is useful for phrase-based retrieval and
    it also gives us term frequencies and document frequencies for free. The fact that
    we derive the simpler inverted index from this structure ensures both indexes stay
    consistent by construction.

    Returns:
        dict term -> {"df": int, "postings": {docid: {"tf": int, "positions": [int,...]}}}
    """
    index = defaultdict(lambda: {"df": 0, "postings": {}})

    for docid, tokens in corpus.doc_tokens.items():
        # Map each term to the positions where it occurs in this document.
        term_positions = defaultdict(list)
        for pos, term in enumerate(tokens):
            term_positions[term].append(pos)

        # Save the per-document term occurrence details into the positional postings.
        for term, positions in term_positions.items():
            postings = index[term]["postings"]
            postings[docid] = {"tf": len(positions), "positions": positions}
            index[term]["df"] += 1

    return dict(index)


def build_inverted_index_from_positional(positional_index):
    """Collapse the positional index into the simpler term-frequency inverted index.

    Part A/B only needs the term frequency per document, not the exact token positions.
    By deriving this structure from the positional index, we guarantee the two remain
    synchronized even if the corpus or preprocessing changes later.
    """
    inverted = {}
    for term, entry in positional_index.items():
        # Each posting keeps only the document frequency and the raw tf count, since
        # the positional detail is not needed for standard BM25 or tf-idf retrieval.
        postings = {docid: p["tf"] for docid, p in entry["postings"].items()}
        inverted[term] = {"df": entry["df"], "postings": postings}
    return inverted


def build_title_field_index(corpus: Corpus):
    """Track title-specific term frequency as a separate field-aware signal.

    This is not part of the base inverted index; it is a specialized index used by the
    fielded title-boost novelty feature. It tells us how often a term appears in a
    document's title (plus category text), which can be used to increase ranking for
    title-heavy matches without altering the main retrieval pipeline.
    """
    title_index = defaultdict(dict)
    for docid, tokens in corpus.title_tokens.items():
        counts = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        for t, c in counts.items():
            title_index[t][docid] = c
    return dict(title_index)


def doc_lengths(corpus: Corpus):
    """Return the length of each document in token count for BM25 and other length-aware scoring.

    Document length is a key normalization signal: longer documents naturally contain more
    terms, so we often need to adjust scores by how much text a document contains.
    """
    return {docid: len(tokens) for docid, tokens in corpus.doc_tokens.items()}
