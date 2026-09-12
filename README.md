# Swatchbook_IR — Clothing Information Retrieval Engine

**CSD358 · Information Retrieval · Assignment 1**

A search engine over a 100-document clothing product catalog, built from
scratch (no external IR libraries) around an inverted index, a Vector
Space Model with `lnc.ltc` weighting, and a positional index for
phrase/proximity search — plus eight additional novelty features layered
on top of that required core.

Two interfaces ship with the same underlying logic:

- **`swatchbook.html`** — a single-file, dependency-free web app. Open it
  directly in a browser; everything (indexing, ranking, search) runs
  client-side in JavaScript.
- **`cli.py`** — a terminal application using the equivalent Python
  backend in `src/`.

Both interfaces are deliberately kept in sync: the same algorithms,
same defaults, same output for the same query.

---

## What's implemented

### Required (Parts A–E)
- **Part A** — tokenization, case normalization, punctuation stripping,
  documented stop-word policy, Porter stemming (implemented from
  scratch), and an inverted index (`term → df → postings[(docID, tf)]`).
- **Part B** — ranked retrieval using the Vector Space Model with
  `lnc.ltc` weighting (log-tf, no-idf, cosine-normalized documents;
  log-tf, idf, cosine-normalized queries), returning the top 10 by
  cosine similarity.
- **Part C** — a positional index (`term → df → postings[(docID, tf,
  [positions])]`), exact phrase search, and ordered proximity
  (`WITHIN/k`) search.
- **Part D** — the two application interfaces above.
- **Part E** — a test harness (`tests/run_tests.py`) running ≥10
  free-text queries, ≥5 phrase queries, ≥3 proximity queries at
  different `k`, an out-of-vocabulary query, and a written explanation
  of ≥2 cases where positional information changes the result
  set/order — all against live query results, nothing hard-coded.

### Novelty features
1. **BM25** ranking, run alongside `lnc.ltc` for comparison.
2. **Sequential Dependence Model (SDM)** — folds ordered/unordered
   positional evidence directly into the ranking score instead of only
   using it for Boolean phrase filtering.
3. **Title/Category field-boosted ranking** (BM25F-inspired) — rewards a
   query term appearing in a product's title/category over the same
   term buried in body text.
4. **Typo-tolerant fallback** — Levenshtein-distance correction for
   out-of-vocabulary query terms.
5. **Result snippets** — a Google-style "best window" excerpt of the
   product description with matched terms highlighted.
6. **Rank-shift badges** — on phrase/proximity results, shows how many
   places a document moved compared to where plain VSM would have
   ranked the same query text (`▲N` / `▼N` / `=` / `NEW`).
7. **NDCG evaluation** — grade your own relevance judgments for a query
   and see NDCG@10/@5 compared live across every ranker.
8. **Custom SMART-notation VSM builder** — compose any `tf.df.norm`
   weighting scheme (not just `lnc.ltc`) for the document side and
   query side independently, and search with it.

See `DOCUMENTATION.md` for how each of these actually works and what
happens, step by step, when a search is run.

---

## Repository structure

```
ir_clothing_search/
├── corpus_100.txt            # the 100-document clothing corpus
├── swatchbook.html           # web interface (self-contained, no build step)
├── cli.py                    # terminal interface
├── dump_indexes.py           # writes the required index deliverables
├── src/
│   ├── preprocess.py         # tokenize / stopword / stem pipeline
│   ├── porter_stemmer.py     # from-scratch Porter stemmer
│   ├── index_builder.py      # inverted index + positional index construction
│   ├── vsm.py                 # Part B: lnc.ltc Vector Space Model
│   ├── positional_search.py  # Part C: phrase + proximity search
│   ├── bm25.py                # novelty: BM25 ranking
│   ├── sdm.py                 # novelty: Sequential Dependence Model
│   ├── fielded.py             # novelty: title/category boosted ranking
│   ├── fuzzy.py                # novelty: typo-tolerant correction
│   ├── snippets.py            # novelty: highlighted snippet generation
│   ├── rank_shift.py          # novelty: rank-shift badges vs. plain VSM
│   ├── ndcg.py                 # novelty: NDCG@k computation
│   ├── smart_vsm.py           # novelty: generalized SMART-notation VSM
│   └── search_engine.py       # facade tying everything together
├── tests/
│   └── run_tests.py           # Part E test harness
└── output/
    ├── inverted_index.txt     # dictionary/inverted-index deliverable
    ├── positional_index.txt   # positional-index deliverable
    └── test_results.md        # Part E test report
```

## Running it

**Web interface** — no installation needed:
```
open swatchbook.html      # or double-click it / drag into a browser
```

**CLI** (Python 3, standard library only — no `pip install` required):
```
python3 cli.py
```

**Regenerate the index deliverables:**
```
python3 dump_indexes.py
```

**Run the Part E test suite:**
```
python3 -m tests.run_tests
```
(writes `output/test_results.md`)

## Design principles we tried to hold to

- **No hard-coded "correct" document IDs anywhere.** Every reported
  result — in the tests, the CLI, or the web demo — comes from
  actually running the query against the live index.
- **Honesty about approximations.** Anywhere we simplified something
  (e.g. the SMART `u`/pivoted-unique normalization, or the rule-based
  proxy relevance judgments used to demonstrate NDCG), it's flagged
  directly in a code comment and in the corresponding UI/report text,
  not left implicit.
- **Zero third-party dependencies.** Both interfaces run with nothing
  beyond the Python standard library / plain browser JavaScript, so a
  grader can run this on any machine without an install step.
