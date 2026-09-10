#!/usr/bin/env python3
"""Export the DeepSeek V4 reasonings into a single JSON artifact.

Reads every reasoning attributed to the deepseek `cot` agent and writes
`results/deepseek/reasonings_deepseek.json` with one entry per reasoning:

    {sentence_id, label_id, run_id, raw_response, parsed, parse_status}

`parsed` is preserved verbatim (not re-extracted) so a later import can
round-trip the reasonings byte-for-byte. The producer model is not stored in
the artifact; attribution is fixed by the import script to the deepseek agent.

Run inside the app container:
    docker compose exec -T -e PYTHONPATH=/app app python scripts/export_deepseek.py
"""

import json
import os

from app.db import pool
from app import agents

DEEPSEEK_MODEL = agents.DEEPSEEK_MODEL
OUT_PATH = "results/deepseek/reasonings_deepseek.json"


def main():
    pool.open()
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT r.sentence_id, r.label_id, r.run_id,
                       r.raw_response, r.parsed, r.parse_status
                FROM reasonings r
                JOIN agents a ON a.id = r.agent_id
                JOIN models m ON m.id = a.model_id
                WHERE m.name = %s
                ORDER BY r.id
                """,
                (DEEPSEEK_MODEL,),
            ).fetchall()
    finally:
        pool.close()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    entries = [
        {
            "sentence_id": r[0],
            "label_id": r[1],
            "run_id": r[2],
            "raw_response": r[3],
            "parsed": r[4],
            "parse_status": r[5],
        }
        for r in rows
    ]
    with open(OUT_PATH, "w") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(f"deepseek reasonings exported: {len(entries)}")
    print(f"artifact: {OUT_PATH}")


if __name__ == "__main__":
    main()