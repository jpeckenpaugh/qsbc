"""Model / Agent / Persona layering.

A **Model** is the base LLM (e.g. `opencode-go/deepseek-v4-flash`) with static
characteristics. An **Agent** is one Model instantiated with a scope and runtime
settings (temperature / seed / variant); it also carries a flexible **persona**
JSON. A reasoning points at an Agent, so the producer is a join
`Reasoning -> Agent -> Model`.

This module owns the `models` + `agents` tables, idempotent seeding of the two
known producers, and the in-place migration of `reasonings.model`
(free-text) -> `reasonings.agent_id` (FK). It must be safe to run repeatedly and
must never destroy source truth (`reasonings`).
"""

from psycopg.types.json import Jsonb

# ------------------------------------------------------------ constants ----

DEEPSEEK_MODEL = "opencode-go/deepseek-v4-flash"
GEMMA4_MODEL = "gemma4:26b-a4b-it-q4_K_M"

COT_AGENT_NAME = "cot"
COT_TEMPERATURE = 0.05
COT_VARIANT = "low"

# The persona, derived from .opencode/agents/cot.md (`description` front-matter).
COT_DESCRIPTION = (
    "You are a C3PA expert. You generate chain-of-thought reasoning statements "
    "explaining a C3PA sentence\u2192label assignment. You produce only a single "
    "JSON array as output, containing 2 - 5 sentences."
)

COT_PERSONA = {
    "role": "C3PA expert",
    "style": "generalized chain-of-thought reasoning",
    "variant": COT_VARIANT,
    "description": COT_DESCRIPTION,
}

# (name, family, size, quantization, source, checkpoint) -- NULLs where unknown.
SEED_MODELS = [
    (DEEPSEEK_MODEL, None, None, None, None, None),
    (GEMMA4_MODEL, "gemma4", "~25.8B", "Q4_K_M", "Ollama", None),
]

SEED_MODEL_NAMES = [m[0] for m in SEED_MODELS]

# ---------------------------------------------------------------- schema ----

SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS models (
        id           BIGSERIAL PRIMARY KEY,
        name         TEXT NOT NULL UNIQUE,
        family       TEXT,
        size         TEXT,
        quantization TEXT,
        source       TEXT,
        checkpoint   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agents (
        id          BIGSERIAL PRIMARY KEY,
        name        TEXT NOT NULL,
        model_id    BIGINT NOT NULL REFERENCES models(id),
        temperature REAL,
        seed        TEXT,
        variant     TEXT,
        scope       TEXT,
        persona     JSONB,
        UNIQUE (name, model_id, temperature, variant)
    )
    """,
    "CREATE INDEX IF NOT EXISTS agents_model_id_idx ON agents (model_id)",
]


def seed(conn):
    """Idempotently insert the known models and their `cot` agents."""
    for name, family, size, quant, source, checkpoint in SEED_MODELS:
        conn.execute(
            """
            INSERT INTO models (name, family, size, quantization, source, checkpoint)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING
            """,
            (name, family, size, quant, source, checkpoint),
        )
    conn.execute(
        """
        INSERT INTO agents (name, model_id, temperature, variant, persona)
        SELECT %s, m.id, %s, %s, %s::jsonb
        FROM models m
        WHERE m.name = ANY(%s)
        ON CONFLICT (name, model_id, temperature, variant) DO NOTHING
        """,
        (COT_AGENT_NAME, COT_TEMPERATURE, COT_VARIANT,
         Jsonb(COT_PERSONA), SEED_MODEL_NAMES),
    )


def resolve_cot_agent(conn, model):
    """Resolve-or-create the `cot` agent for a model name; return agent_id.

    Used by POST /api/reasonings so the backward-compatible `model` string in
    the request body keeps working. Creates the model row (if unknown) and its
    `cot` agent. Returns None only when `model` is falsy."""
    if not model:
        return None
    conn.execute(
        "INSERT INTO models (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
        (model,),
    )
    model_id = conn.execute(
        "SELECT id FROM models WHERE name = %s", (model,)
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO agents (name, model_id, temperature, variant, persona)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (name, model_id, temperature, variant) DO NOTHING
        """,
        (COT_AGENT_NAME, model_id, COT_TEMPERATURE, COT_VARIANT, Jsonb(COT_PERSONA)),
    )
    return conn.execute(
        """
        SELECT id FROM agents
        WHERE name = %s AND model_id = %s AND temperature = %s::real AND variant = %s
        """,
        (COT_AGENT_NAME, model_id, COT_TEMPERATURE, COT_VARIANT),
    ).fetchone()[0]


def migrate_reasonings(conn):
    """In-place, idempotent migration of reasonings.model -> agent_id.

    Order: add nullable agent_id -> backfill via (model, run_id) -> agent ->
    FK + NOT NULL -> drop `model`. Guarded so it is safe to run repeatedly even
    after `model` has been dropped."""
    conn.execute("ALTER TABLE reasonings ADD COLUMN IF NOT EXISTS agent_id BIGINT")

    # Backfill only while the free-text `model` column still exists.
    has_model_col = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'reasonings' AND column_name = 'model'"
    ).fetchone()
    if has_model_col:
        conn.execute(
            """
            UPDATE reasonings r
            SET agent_id = a.id
            FROM agents a
            JOIN models m ON m.id = a.model_id
            WHERE r.agent_id IS NULL
              AND a.name = %s
              AND m.name = r.model
            """,
            (COT_AGENT_NAME,),
        )

    conn.execute(
        """
        DO $mig$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint
                           WHERE conname = 'reasonings_agent_id_fkey') THEN
                ALTER TABLE reasonings ADD CONSTRAINT reasonings_agent_id_fkey
                    FOREIGN KEY (agent_id) REFERENCES agents(id);
            END IF;
        END $mig$
        """
    )

    nulls = conn.execute(
        "SELECT count(*) FROM reasonings WHERE agent_id IS NULL"
    ).fetchone()[0]
    if nulls:
        raise RuntimeError(
            f"{nulls} reasonings still lack an agent_id; refusing to SET NOT NULL"
        )
    conn.execute("ALTER TABLE reasonings ALTER COLUMN agent_id SET NOT NULL")

    conn.execute("ALTER TABLE reasonings DROP COLUMN IF EXISTS model")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS reasonings_agent_id_idx ON reasonings (agent_id)"
    )


def ensure_schema(conn):
    for ddl in SCHEMA_DDL:
        conn.execute(ddl)
    seed(conn)
    migrate_reasonings(conn)