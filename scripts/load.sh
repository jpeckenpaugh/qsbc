#!/usr/bin/env bash
# Import the parsed C3PA TSV files into Postgres.
# Designed to run inside the app container (psql client + PGHOST env),
# e.g.: docker compose exec app bash scripts/load.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PGHOST="${PGHOST:-postgres}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-c3pa}"
PGDATABASE="${PGDATABASE:-c3pa}"

PSQL=(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1)

"${PSQL[@]}" -f db/init/01_schema.sql >/dev/null

echo "Importing labels..."
"${PSQL[@]}" -c "TRUNCATE labels CASCADE; COPY labels (id, name) FROM STDIN" < data/labels.tsv

echo "Importing documents..."
"${PSQL[@]}" -c "TRUNCATE documents CASCADE; COPY documents (id, subset, doc_num, source_file, doc_key, url) FROM STDIN" < data/documents.tsv

echo "Importing sentences..."
"${PSQL[@]}" -c "TRUNCATE sentences CASCADE; COPY sentences (id, doc_id, text) FROM STDIN" < data/sentences.tsv

echo "Importing sentence_labels..."
"${PSQL[@]}" -c "TRUNCATE sentence_labels; COPY sentence_labels (sentence_id, label_id) FROM STDIN" < data/sentence_labels.tsv

echo
"${PSQL[@]}" -c "SELECT (SELECT count(*) FROM documents) AS documents,
                        (SELECT count(*) FROM sentences) AS sentences,
                        (SELECT count(*) FROM labels)     AS labels,
                        (SELECT count(*) FROM reasonings) AS reasonings;"