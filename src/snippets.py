"""
NOVELTY -- backend port of swatchbook.html's buildSnippet(): a Google-
style "best window" snippet. Slides a fixed-size window of stemmed-token
positions over a document, picks the window with the most hits against
the query's stemmed terms, then renders the ORIGINAL text in that
window with matches marked. Falls back to the start of the document if
nothing matches, so results never look bare.
"""

from src.preprocess import preprocess_with_spans


def _default_highlight(raw_text):
    return f"**{raw_text}**"


def build_snippet(corpus, docid, query_terms_stemmed, window_tokens=16, highlight_fn=None):
    """
    corpus: an index_builder.Corpus instance.
    query_terms_stemmed: iterable of already-stemmed query terms (e.g.
        from src.preprocess.preprocess(query)).
    highlight_fn: how to mark a matched raw token; defaults to Markdown
        bold (renders in the Part E .md report and is still readable as
        plain text). Pass e.g. an ANSI-wrapping function for a nicer
        interactive terminal look.
    """
    highlight_fn = highlight_fn or _default_highlight
    doc = corpus.doc_by_id[docid]
    full_text = doc["title"] + ". " + doc["text"]
    spans = preprocess_with_spans(full_text)
    if not spans:
        return full_text[:140]

    q_set = set(query_terms_stemmed or [])

    best_start, best_end, best_hits = 0, min(window_tokens, len(spans)), -1
    for start in range(len(spans)):
        end = min(start + window_tokens, len(spans))
        hits = sum(1 for i in range(start, end) if spans[i]["stem"] in q_set)
        if hits > best_hits:
            best_start, best_end, best_hits = start, end, hits
        if end >= len(spans):
            break

    pad = 3  # a little context so we don't cut mid-punctuation
    char_start = max(0, spans[best_start]["start"] - pad)
    char_end = min(len(full_text), spans[best_end - 1]["end"] + pad)

    out = []
    cursor = char_start
    for i in range(best_start, best_end):
        s = spans[i]
        if s["start"] < char_start or s["end"] > char_end:
            continue
        out.append(full_text[cursor:s["start"]])
        raw = full_text[s["start"]:s["end"]]
        out.append(highlight_fn(raw) if s["stem"] in q_set else raw)
        cursor = s["end"]
    out.append(full_text[cursor:char_end])

    prefix = "\u2026" if char_start > 0 else ""
    suffix = "\u2026" if char_end < len(full_text) else ""
    return prefix + "".join(out) + suffix
