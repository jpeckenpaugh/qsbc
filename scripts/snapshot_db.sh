#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE="docker compose"

if ! $COMPOSE ps --status running | grep -qE 'c3pa-(postgres|app)'; then
  echo "Error: the docker compose services (postgres/app) are not running." >&2
  echo "Start them first with: docker compose up -d" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
mkdir -p snapshots
OUT="snapshots/snapshot_${STAMP}.sql.gz"

if ! $COMPOSE exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' 2>snapshots/.dump_err | gzip > "$OUT"; then
  echo "Error: pg_dump failed." >&2
  cat snapshots/.dump_err >&2 2>/dev/null || true
  rm -f "$OUT"
  exit 1
fi
rm -f snapshots/.dump_err

echo "Snapshot written: $OUT"
ls -lh "$OUT"