#!/usr/bin/env python3
"""Parse the C3PA dataset into TSV files for PostgreSQL import.

Source layout (C3PA_Dataset/):
  Annotations/DB/1.csv ... 230.csv      - Data Broker annotation files
  Annotations/WS/1.csv ... 170.csv      - Website annotation files
  Crawl/db.csv, Crawl/ws.csv            - document URLs (best effort)

Each annotation row is (RANumb, Text, Label). Passages are normalized
(literal \n escapes -> newlines, \xa0 -> space), segmented into sentences,
and deduplicated per document. A sentence may carry multiple labels.
"""

import csv
import glob
import os
import re
import sys

REPO_ROOT = "C3PA_Dataset"
OUT_DIR = "data"

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
MIN_WORDS = 4  # drop sentence fragments / table artifacts

SUBSETS = [("DB", 230), ("WS", 170)]


def esc_field(value):
    """Escape backslashes so PostgreSQL COPY (text format) round-trips verbatim.
    COPY interprets \\n, \\r, \\t, \\\\ as escapes, so a literal backslash-n in the
    TSV would otherwise be imported as a real newline."""
    return value.replace("\\", "\\\\")


def load_crawl_urls(subset):
    """Best-effort URL lookup: Crawl/{subset}.csv row i -> annotation file i+1."""
    path = os.path.join(REPO_ROOT, "Crawl", f"{subset.lower()}.csv")
    urls = {}
    if not os.path.exists(path):
        print(f"  [warn] no crawl file {path}; url left NULL")
        return urls
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            urls[i + 1] = row.get("Link", "").strip() or None
    return urls


def normalize(text):
    """Keep the notebook's exact pipeline: no literal '\\n' handling.
    (Some source files contain backslash-n escapes; the notebook treated them
    as literal text, so we do too, to reproduce the published 37,284 count.
    str.split() already collapses \\xa0 and real whitespace below.)"""
    return text.strip()


def parse_document(path, doc_num, url):
    """Return list of (sentence_text, label) for one annotation file."""
    pairs = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        col_map = {c.lower(): c for c in (reader.fieldnames or [])}
        if "text" not in col_map or "label" not in col_map:
            return pairs
        tcol, lcol = col_map["text"], col_map["label"]
        for row in reader:
            text = (row.get(tcol) or "").strip()
            label = (row.get(lcol) or "").strip()
            if not text or not label or label.lower() == "nan":
                continue
            for sentence in SENTENCE_SPLIT_REGEX.split(normalize(text)):
                sentence = " ".join(sentence.split())
                if len(sentence.split()) >= MIN_WORDS:
                    pairs.append((sentence, label))
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_labels = set()
    documents = []      # (subset, doc_num, source_file, doc_key, url)
    sentences = []      # (doc_id, text) -> sentence_id
    sentence_ids = {}   # (doc_id, text) -> id
    sentence_labels = []  # (sentence_id, label_id)

    sentence_id = 0

    for subset, count in SUBSETS:
        urls = load_crawl_urls(subset)
        for doc_num in range(1, count + 1):
            path = os.path.join(REPO_ROOT, "Annotations", subset, f"{doc_num}.csv")
            if not os.path.exists(path):
                print(f"  [warn] missing {path}")
                continue
            doc_id = len(documents) + 1
            doc_key = f"{subset}_{doc_num}"
            source_file = f"{subset}/{doc_num}.csv"
            url = urls.get(doc_num)
            documents.append((subset, doc_num, source_file, doc_key, url))

            seen = set()
            for text, label in parse_document(path, doc_num, url):
                all_labels.add(label)
                key = (doc_id, text)
                if key not in sentence_ids:
                    sentence_id += 1
                    sentence_ids[key] = sentence_id
                    sentences.append((sentence_id, doc_id, text))
                pair = (sentence_ids[key], label)
                if pair not in seen:
                    seen.add(pair)
                    sentence_labels.append(pair)

    # Resolve label names -> ids
    labels = sorted(all_labels)
    label_ids = {name: i + 1 for i, name in enumerate(labels)}

    # Write TSVs
    with open(os.path.join(OUT_DIR, "labels.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for name in labels:
            w.writerow([label_ids[name], name])

    with open(os.path.join(OUT_DIR, "documents.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for i, (subset, doc_num, source_file, doc_key, url) in enumerate(documents, start=1):
            w.writerow([i, subset, doc_num, source_file, doc_key, esc_field(url or "")])

    with open(os.path.join(OUT_DIR, "sentences.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for sid, doc_id, text in sentences:
            w.writerow([sid, doc_id, esc_field(text)])

    with open(os.path.join(OUT_DIR, "sentence_labels.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for sid, label in sentence_labels:
            w.writerow([sid, label_ids[label]])

    # Stats
    from collections import Counter
    per_sentence = Counter()
    sl_by_sent = {}
    for sid, label in sentence_labels:
        sl_by_sent.setdefault(sid, set()).add(label)
    for sid, labs in sl_by_sent.items():
        per_sentence[len(labs)] += 1

    print(f"documents           : {len(documents):,}  ({SUBSETS})")
    print(f"labels              : {len(labels)}")
    print(f"sentences (unique)  : {len(sentences):,}")
    print(f"sentence_labels     : {len(sentence_labels):,}")
    print(f"multi-label counts  : {dict(sorted(per_sentence.items()))}")
    print(f"TSV output written to {OUT_DIR}/")


if __name__ == "__main__":
    main()