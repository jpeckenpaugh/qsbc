#!/usr/bin/env python3
"""Import gemma4 reasonings from results/gemma4/*.json into the DB.

The gemma4 JSON entries have NO `parsed` field, so this script re-runs
`extract_json_array(raw_response)` to populate `parsed` (status `ok`), or sets
`parsed = NULL` / status `unparseable` for the 2 unparseable responses (matches
the reasonings CHECK constraint). Each reasoning is attributed to the gemma4
`cot` agent under run `colab-gemma4-01`.

Dedup is on the FULL (sentence_id, label_id, run_id) triple via ON CONFLICT, so
gemma4 rows that share a sentence+label with a deepseek row (but a different
run) are still inserted. Source reasonings are never touched.

Run inside the app container:
    docker compose exec -T -e PYTHONPATH=/app app python scripts/import_gemma4.py
"""

import glob
import json

from psycopg.types.json import Jsonb

from app.db import pool
from app.extract import extract_json_array
from app import agents, thoughts

GEMMA4_RUN = "colab-gemma4-01"


def main():
    files = sorted(glob.glob("results/gemma4/*.json"))
    if not files:
        raise SystemExit("no gemma4 files found under results/gemma4/")
    total = sum(len(json.load(open(f))) for f in files)

    pool.open()
    try:
        # Ensure models/agents exist and the gemma4 cot agent is seeded BEFORE
        # importing (and the reasonings migration is applied).
        with pool.connection() as conn:
            agents.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT a.id FROM agents a
                JOIN models m ON m.id = a.model_id
                WHERE a.name = %s AND m.name = %s
                """,
                (agents.COT_AGENT_NAME, agents.GEMMA4_MODEL),
            ).fetchone()
            if row is None:
                raise SystemExit("gemma4 cot agent not found after ensure_schema")
            gemma4_agent_id = row[0]

        inserted = skipped = unparseable = 0
        for fn in files:
            for e in json.load(open(fn)):
                sid, lid = e["sentence_id"], e["label_id"]
                raw = e["raw_response"]
                if e.get("parse_status") == "unparseable":
                    status, parsed = "unparseable", None
                    unparseable += 1
                else:
                    status, parsed = extract_json_array(raw)
                with pool.connection() as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO reasonings
                            (sentence_id, label_id, run_id, agent_id, raw_response, parsed, parse_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sentence_id, label_id, run_id) DO NOTHING
                        RETURNING id
                        """,
                        (sid, lid, GEMMA4_RUN, gemma4_agent_id, raw,
                         Jsonb(parsed) if parsed is not None else None, status),
                    )
                    if cur.rowcount:
                        inserted += 1
                        if parsed:
                            thoughts.insert_for_reasoning(
                                conn, cur.fetchone()[0], sid, lid, GEMMA4_RUN, parsed
                            )
                    else:
                        skipped += 1

        with pool.connection() as conn:
            now = conn.execute(
                "SELECT count(*) FROM reasonings WHERE run_id = %s", (GEMMA4_RUN,)
            ).fetchone()[0]
        print(f"gemma4 source entries: {total}")
        print(f"inserted: {inserted}")
        print(f"skipped (already present): {skipped}")
        print(f"unparseable: {unparseable}")
        print(f"reasonings now on run {GEMMA4_RUN}: {now}")
    finally:
        pool.close()


if __name__ == "__main__":
    main()