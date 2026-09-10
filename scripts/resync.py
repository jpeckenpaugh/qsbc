"""Rebuild derived entities from the source of truth.

Source of truth = `reasonings` (raw LLM outputs). Everything below it is
derived and recomputable:

  1. Thoughts  <- normalize each parsed statement of every reasoning.
  2. Ideas     <- cluster the distinct normalized thoughts.

This drops and regenerates `thoughts`, `thought_idea`, `ideas`, and drops the
orphaned pre-rename tables (`claims`, `claims_sync`, `claim_clusters`,
`claim_norm_cluster`). Reasonings are never touched.

Run inside the app container: docker compose exec app python scripts/resync.py
"""

from app.db import pool
from app import thoughts, ideas

THRESHOLD = 0.8
BATCH_SIZE = 1000

ORPHAN_TABLES = ["claims", "claims_sync", "claim_clusters", "claim_norm_cluster"]


def main():
    pool.open()
    try:
        with pool.connection() as conn:
            thoughts.ensure_schema(conn)
            ideas.ensure_schema(conn)
            # Reset derived tables + watermark (idempotent).
            conn.execute(
                "TRUNCATE thought_idea, ideas, thoughts, thoughts_sync "
                "RESTART IDENTITY CASCADE"
            )
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

        # 2. Ideas: cluster distinct thought norms.
        with pool.connection() as conn:
            norms = [
                x[0]
                for x in conn.execute("SELECT DISTINCT norm FROM thoughts").fetchall()
            ]
        print(f"distinct thought norms: {len(norms)}")
        if not norms:
            raise SystemExit("no thoughts to cluster; aborting before dropping old tables")
        ideas_list, failed = ideas.build_ideas(norms, threshold=THRESHOLD)
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