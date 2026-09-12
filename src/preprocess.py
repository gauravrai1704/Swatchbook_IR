"""
Part A: Corpus parsing and pre-processing pipeline.

Pipeline for every document / query:
    1. Tokenize on non-alphanumeric boundaries.
    2. Lowercase (case normalization).
    3. Strip punctuation (done implicitly by the tokenizer regex).
    4. Remove stop-words.
    5. Apply Porter stemming.

Stop-word policy:
    We use a standard, fairly small English stop-word list (the classic
    "SMART" / Van Rijsbergen 25-ish core function-word list, trimmed to
    the words that actually occur in this corpus's templated sentences).
    Every product description in this corpus is generated from a fixed
    template ("Made from X, this Y is designed for everyday Indian wear.
    It features A, B, and C. The Z colour works well for casual, office,
    travel, or festive styling depending on the garment. Available in
    size ... The garment is suitable for comfortable regular use ...").
    That template contributes a large number of high-document-frequency
    function words and template-filler words (e.g. "the", "is", "for",
    "and", "this", "it", "well", "depending") that occur in nearly all
    100 documents and therefore have almost no discriminative power in
    the VSM (idf ~ 0) -- but WOULD be dead weight in the positional
    index (huge postings lists) and can distort proximity windows.
    We remove standard English function words. We deliberately do NOT
    stem/remove domain words that happen to be frequent (e.g. "cotton",
    "wear", "fit") -- df alone does not justify removing a content word,
    only closed-class function words are stop-worded, applied
    consistently across corpus and queries alike.
"""

import re
import os

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "depending",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "other", "such", "than", "that", "the", "this", "to",
    "well", "with", "works", "made", "designed", "available", "suitable",
    "paired", "common", "features",
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text):
    """Lowercase + split on non-alphanumeric characters (strips punctuation)."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def preprocess(text, keep_positions=False):
    """
    Full pipeline: tokenize -> normalize case -> remove stopwords -> stem.

    Returns a list of stemmed tokens. If keep_positions is True, the
    returned tokens still line up 1:1 with sequential position numbers
    (stopword removal shifts positions the same way for indexing and for
    queries, so phrase/proximity search stays consistent as long as BOTH
    documents and queries go through this same function).
    """
    from src.porter_stemmer import porter_stem
    raw_tokens = tokenize(text)
    tokens = [porter_stem(t) for t in raw_tokens if t not in STOPWORDS]
    return tokens


def preprocess_with_spans(text):
    """
    NOVELTY (backend port of swatchbook.html's preprocessWithSpans) --
    same tokenize -> stopword-filter -> stem pipeline as preprocess(),
    but each surviving token also keeps its raw text and character span
    in the ORIGINAL string. Element i here lines up 1:1 with position i
    in preprocess(text) (both apply the identical filter), which lets
    src/snippets.py map a stemmed-token match -- including positions
    coming straight out of the positional index -- back to highlightable
    original text without any separate offset bookkeeping.

    Returns a list of dicts: {"stem", "raw", "start", "end"}.
    """
    from src.porter_stemmer import porter_stem
    spans = []
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0)
        lower = raw.lower()
        if lower in STOPWORDS:
            continue
        spans.append({
            "stem": porter_stem(lower),
            "raw": raw,
            "start": m.start(),
            "end": m.end(),
        })
    return spans


DOC_RE = re.compile(
    r"<DOC>\s*<DOCID>(.*?)</DOCID>\s*<CATEGORY>(.*?)</CATEGORY>\s*"
    r"<TITLE>(.*?)</TITLE>\s*<TEXT>(.*?)</TEXT>\s*</DOC>",
    re.DOTALL,
)


def parse_corpus(path):
    """
    Parses the pipe corpus format:
        <DOC><DOCID>..</DOCID><CATEGORY>..</CATEGORY><TITLE>..</TITLE><TEXT>..</TEXT></DOC>

    Returns a list of dicts: {docid, category, title, text}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    docs = []
    for m in DOC_RE.finditer(raw):
        docid, category, title, text = m.groups()
        docs.append({
            "docid": docid.strip(),
            "category": category.strip(),
            "title": title.strip(),
            "text": text.strip(),
        })
    return docs
