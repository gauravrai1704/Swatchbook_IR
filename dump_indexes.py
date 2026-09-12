"""Dumps the inverted index and positional index to output/ as required deliverables."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.search_engine import SearchEngine

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_100.txt")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def dump_inverted(engine, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"INVERTED INDEX  (N={engine.corpus.N} documents, "
                f"{len(engine.inverted_index)} unique terms)\n")
        f.write("Format: term  df  -> [(docID, tf), ...]\n")
        f.write("=" * 70 + "\n")
        for term in sorted(engine.inverted_index.keys()):
            entry = engine.inverted_index[term]
            postings = sorted(entry["postings"].items(), key=lambda x: x[0])
            postings_str = ", ".join(f"({d},{tf})" for d, tf in postings)
            f.write(f"{term:<15} df={entry['df']:<4} -> [{postings_str}]\n")


def dump_positional(engine, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"POSITIONAL INDEX  (N={engine.corpus.N} documents, "
                f"{len(engine.positional_index)} unique terms)\n")
        f.write("Format: term  df  -> [(docID, tf, [positions]), ...]\n")
        f.write("=" * 70 + "\n")
        for term in sorted(engine.positional_index.keys()):
            entry = engine.positional_index[term]
            postings = sorted(entry["postings"].items(), key=lambda x: x[0])
            parts = []
            for d, p in postings:
                parts.append(f"({d},{p['tf']},{p['positions']})")
            f.write(f"{term:<15} df={entry['df']:<4} ->\n    [" + ",\n     ".join(parts) + "]\n\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = SearchEngine(CORPUS_PATH)
    dump_inverted(engine, os.path.join(OUT_DIR, "inverted_index.txt"))
    dump_positional(engine, os.path.join(OUT_DIR, "positional_index.txt"))
    print("Wrote output/inverted_index.txt and output/positional_index.txt")


if __name__ == "__main__":
    main()
