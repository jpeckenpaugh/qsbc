"""Reasoning-thought storage, and sync/backfill logic.

A "reasoning" is a collection of statements an agent produces explaining why a
sentence carries a label. Each individual statement is a "thought". This module
maintains a `thoughts` table plus a `thoughts_sync` watermark so we can tell
when new reasonings are out of sync with their extracted thoughts.
"""

# ----------------------------------------------------------------- schema ----

SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS thoughts (
        id           BIGSERIAL PRIMARY KEY,
        reasoning_id INTEGER NOT NULL REFERENCES reasonings(id) ON DELETE CASCADE,
        sentence_id  INTEGER NOT NULL,
        label_id     INTEGER NOT NULL,
        run_id       TEXT NOT NULL,
        pos          INTEGER NOT NULL,
        text         TEXT NOT NULL,
        UNIQUE (reasoning_id, pos)
    )
    """,
    "CREATE INDEX IF NOT EXISTS thoughts_reasoning_id_idx ON thoughts (reasoning_id)",
    "CREATE INDEX IF NOT EXISTS thoughts_run_id_idx ON thoughts (run_id)",
    "CREATE INDEX IF NOT EXISTS thoughts_label_id_idx ON thoughts (label_id)",
    """
    CREATE TABLE IF NOT EXISTS thoughts_sync (
        id                INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
        last_reasoning_id BIGINT NOT NULL DEFAULT 0
    )
    """,
    "INSERT INTO thoughts_sync (id) VALUES (1) ON CONFLICT (id) DO NOTHING",
]


def ensure_schema(conn):
    for ddl in SCHEMA_DDL:
        conn.execute(ddl)
    # Idempotent migration from the old norm-keyed schema.
    conn.execute("DROP INDEX IF EXISTS thoughts_norm_idx")
    conn.execute("ALTER TABLE thoughts DROP COLUMN IF EXISTS norm")


def watermark(conn):
    row = conn.execute("SELECT last_reasoning_id FROM thoughts_sync WHERE id = 1").fetchone()
    return row[0] if row else 0


def _set_watermark(conn, reasoning_id):
    conn.execute(
        "UPDATE thoughts_sync SET last_reasoning_id = GREATEST(last_reasoning_id, %s) WHERE id = 1",
        (reasoning_id,),
    )


# ------------------------------------------------------------- insert/backfill ----

def insert_for_reasoning(conn, reasoning_id, sentence_id, label_id, run_id, parsed):
    """Insert thought rows for one parsed reasoning. Caller owns the transaction.

    Returns number of thoughts inserted. Assumes the reasoning row is not yet in
    the thoughts table (UNIQUE (reasoning_id, pos) protects us regardless)."""
    if not parsed:
        return 0
    rows = [
        (reasoning_id, sentence_id, label_id, run_id, i, s)
        for i, s in enumerate(parsed)
    ]
    for row in rows:
        conn.execute(
            """
            INSERT INTO thoughts (reasoning_id, sentence_id, label_id, run_id, pos, text)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (reasoning_id, pos) DO NOTHING
            """,
            row,
        )
    _set_watermark(conn, reasoning_id)
    return len(rows)


def backfill(conn, batch_size=500):
    """Extract thoughts for reasonings newer than the watermark, in batches.

    Runs inside the caller's transaction. Returns
    {processed, inserted, remaining, total_pending}."""
    wm = watermark(conn)
    max_id = conn.execute("SELECT COALESCE(max(id), 0) FROM reasonings").fetchone()[0]
    total_pending = conn.execute(
        "SELECT count(*) FROM reasonings WHERE id > %s AND parse_status = 'ok'", (wm,)
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT id, sentence_id, label_id, run_id, parsed
        FROM reasonings
        WHERE id > %s AND parse_status = 'ok'
        ORDER BY id
        LIMIT %s
        """,
        (wm, batch_size),
    ).fetchall()

    inserted = 0
    last_id = wm
    for r in rows:
        inserted += insert_for_reasoning(
            conn, r[0], r[1], r[2], r[3], r[4]
        )
        last_id = r[0]
    _set_watermark(conn, last_id)

    remaining = conn.execute(
        "SELECT count(*) FROM reasonings WHERE id > %s AND parse_status = 'ok'",
        (watermark(conn),),
    ).fetchone()[0]

    return {
        "processed": len(rows),
        "inserted": inserted,
        "remaining": remaining,
        "total_pending": total_pending,
    }