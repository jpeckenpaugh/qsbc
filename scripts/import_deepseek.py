#!/usr/bin/env python3
"""Import DeepSeek V4 reasonings from a JSON artifact into the DB.

Inverse of scripts/export_deepseek.py. Inserts `reasonings` rows ONLY (no
thought rows) so a subsequent `resync.py` regenerates all downstream entities
from scratch — this is the flush/re-import vet path.

The artifact entries carry {sentence_id, label_id, run_id, raw_response,
parsed, parse_status}; `parsed` is preserved verbatim so the round-trip is
byte-for-byte. Attribution is fixed to the deepseek `cot` agent.

Dedup is on the FULL (sentence_id, label_id, run_id) triple via ON CONFLICT.
Expects the reasonings table to be empty (post-flush), so conflicts should be 0.

Run inside the app container:
    docker compose exec -T -e PYTHONPATH=/app app python scripts/import_deepseek.py
"""

import json

from psycopg.types.json import Jsonb

from app.db import pool
from app import agents

ARTIFACT = "results/deepseek/reasonings_deepseek.json"


def main():
    with open(ARTIFACT) as fh:
        entries = json.load(fh)

    pool.open()
    try:
        with pool.connection() as conn:
            agents.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT a.id FROM agents a
                JOIN models m ON m.id = a.model_id
                WHERE a.name = %s AND m.name = %s
                """,
                (agents.COT_AGENT_NAME, agents.DEEPSEEK_MODEL),
            ).fetchone()
            if row is None:
                raise SystemExit("deepseek cot agent not found after ensure_schema")
            agent_id = row[0]

        inserted = skipped = 0
        for e in entries:
            status = e["parse_status"]
            parsed = e["parsed"] if status == "ok" else None
            with pool.connection() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO reasonings
                        (sentence_id, label_id, run_id, agent_id, raw_response, parsed, parse_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sentence_id, label_id, run_id) DO NOTHING
                    RETURNING id
                    """,
                    (e["sentence_id"], e["label_id"], e["run_id"], agent_id,
                     e["raw_response"], Jsonb(parsed) if parsed is not None else None,
                     status),
                )
                if cur.rowcount:
                    inserted += 1
                else:
                    skipped += 1

        with pool.connection() as conn:
            total = conn.execute("SELECT count(*) FROM reasonings").fetchone()[0]
        print(f"deepseek source entries: {len(entries)}")
        print(f"inserted: {inserted}")
        print(f"skipped (already present): {skipped}")
        print(f"reasonings total in DB: {total}")
    finally:
        pool.close()


if __name__ == "__main__":
    main()