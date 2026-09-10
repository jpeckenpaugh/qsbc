"""Rebuild derived entities from the source of truth.

Source of truth = `reasonings` (raw LLM outputs). Everything below it is
derived and recomputable:

  1. Thoughts  <- each parsed statement of every reasoning.
  2. Ideas     <- cluster the distinct thought texts.

This drops and regenerates `thoughts`, `thought_idea`, `ideas`, and drops the
orphaned pre-rename tables (`claims`, `claims_sync`, `claim_clusters`,
`claim_norm_cluster`). Reasonings are never touched.

Run inside the app container: docker compose exec app python scripts/resync.py
"""

from app.db import pool
from app import thoughts, ideas, agents

THRESHOLD = 0.8
BATCH_SIZE = 1000

ORPHAN_TABLES = ["claims", "claims_sync", "claim_clusters", "claim_norm_cluster"]

# Derived tables, dropped before schema creation so the new structure (no
# `norm`, `thought_id`-keyed thought_idea, `central` text) is built fresh.
DERIVED_TABLES = ["thought_idea", "ideas", "thoughts", "thoughts_sync"]


def main():
    pool.open()
    try:
        with pool.connection() as conn:
            for t in DERIVED_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            agents.ensure_schema(conn)
            thoughts.ensure_schema(conn)
            ideas.ensure_schema(conn)
            # Reset watermark (idempotent).
            conn.execute(
                "INSERT INTO thoughts_sync (id, last_reasoning_id) VALUES (1, 0) "
                "ON CONFLICT (id) DO UPDATE SET last_reasoning_id = 0"
            )

        # 1. Thoughts: backfill from ALL reasonings (loop until nothing pending).
        total_thoughts = 0
        while True:
            with pool.connection() as conn:
                r = thoughts.backfill(conn, batch_size=BATCH_SIZE)
            total_thoughts += r["inserted"]
            print(f"backfill: processed={r['processed']} inserted={r['inserted']} remaining={r['remaining']}")
            if r["remaining"] == 0:
                break

        # 2. Ideas: cluster distinct thought texts.
        with pool.connection() as conn:
            texts = [
                x[0]
                for x in conn.execute("SELECT DISTINCT text FROM thoughts ORDER BY text").fetchall()
            ]
        print(f"distinct thought texts: {len(texts)}")
        if not texts:
            raise SystemExit("no thoughts to cluster; aborting before dropping old tables")
        ideas_list, failed = ideas.build_ideas(texts, threshold=THRESHOLD)
        print(f"ideas built: {len(ideas_list)} (unembeddable: {failed})")
        with pool.connection() as conn:
            stats = ideas.store_ideas(conn, ideas_list, THRESHOLD)
        print(f"ideas stored: {stats}")

        # 3. Drop orphaned pre-rename tables.
        with pool.connection() as conn:
            for t in ORPHAN_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
                print(f"dropped orphaned table: {t}")

        # 4. Final counts.
        with pool.connection() as conn:
            for t in ["reasonings", "thoughts", "ideas", "thought_idea"]:
                print(f"{t}: {conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]}")
    finally:
        pool.close()


if __name__ == "__main__":
    main()