"""Reasoning-thought normalization, storage, and sync/backfill logic.

A "reasoning" is a collection of statements an agent produces explaining why a
sentence carries a label. Each individual statement is a "thought". This module
normalizes thought text into a canonical key (for overlap/duplicate detection)
and maintains a `thoughts` table plus a `thoughts_sync` watermark so we can tell
when new reasonings are out of sync with their extracted thoughts.
"""

import re
import unicodedata

# ------------------------------------------------------------------ nltk ----

_lemmatizer = None
_lemmatizer_checked = False


def _get_lemmatizer():
    """Lazily load a WordNet lemmatizer. Returns None if unavailable (so the
    pipeline degrades gracefully until the container is rebuilt with nltk)."""
    global _lemmatizer, _lemmatizer_checked
    if not _lemmatizer_checked:
        _lemmatizer_checked = True
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            _lemmatizer = WordNetLemmatizer()
        except Exception:
            _lemmatizer = None
    return _lemmatizer


# Tokenizer: words (lowercased later) or numbers (kept as values).
_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+(?:[.,]\d+)*")


def normalize(thought, lemmatize=True):
    """Return a canonical, comparable form of a thought.

    Pipeline: NFKC -> casefold -> tokenize words/numbers -> strip thousands
    commas -> (optional) lemmatize words -> join with single spaces.

    Meaningful numeric values are preserved (not collapsed to a placeholder),
    so threshold/amount thoughts are not over-merged.
    """
    if not thought:
        return ""
    text = unicodedata.normalize("NFKC", thought).casefold()
    tokens = _TOKEN_RE.findall(text)
    out = []
    for tok in tokens:
        if tok.isdigit() or (tok[0].isdigit()):
            out.append(tok.replace(",", ""))
            continue
        if lemmatize:
            lemmatizer = _get_lemmatizer()
            if lemmatizer is not None:
                tok = lemmatizer.lemmatize(tok, "v")
        out.append(tok)
    return " ".join(out)


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
        norm         TEXT NOT NULL,
        UNIQUE (reasoning_id, pos)
    )
    """,
    "CREATE INDEX IF NOT EXISTS thoughts_norm_idx ON thoughts (norm)",
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
        (reasoning_id, sentence_id, label_id, run_id, i, s, normalize(s))
        for i, s in enumerate(parsed)
    ]
    for row in rows:
        conn.execute(
            """
            INSERT INTO thoughts (reasoning_id, sentence_id, label_id, run_id, pos, text, norm)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
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