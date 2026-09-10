"""Build Ideas by grouping similar Thoughts by semantic similarity.

Approach: embed each distinct thought text with MiniLM, then group thoughts
whose pairwise cosine similarity exceeds a threshold into connected components
(agglomerative). Run as an on-demand background job; results are stored in
`ideas` + `thought_idea` so the UI can show buckets.
"""

import numpy as np

from app import embeddings

# ------------------------------------------------------------------ schema ----

SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ideas (
        id           BIGSERIAL PRIMARY KEY,
        size         INTEGER NOT NULL,
        threshold    REAL NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        central      TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS thought_idea (
        thought_id BIGINT PRIMARY KEY REFERENCES thoughts(id) ON DELETE CASCADE,
        idea_id    BIGINT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS thought_idea_idea_idx ON thought_idea (idea_id)",
]


def ensure_schema(conn):
    for ddl in SCHEMA_DDL:
        conn.execute(ddl)
    # Idempotent migration from the old norm-keyed schema.
    conn.execute("ALTER TABLE ideas DROP COLUMN IF EXISTS central_norm")
    conn.execute("ALTER TABLE thought_idea DROP COLUMN IF EXISTS norm")


# ---------------------------------------------------------------- clustering ----

def build_ideas(texts, threshold=0.8, batch=1000, min_idea_size=2):
    """Embed `texts` and group into Ideas by average-linkage agglomerative clustering.

    Average linkage only merges groups whose *average* pairwise similarity is
    >= threshold, avoiding the single-linkage chaining that collapses thoughts
    sharing a common template opening.

    Returns (ideas, failed) where ideas is a list of (medoid_text, members)
    tuples (each cluster with size >= min_idea_size; members is a list of text
    strings) and failed is a count of unembeddable texts."""
    if not texts:
        return [], 0
    vecs = embeddings.encode_batch(texts)
    if vecs is None:
        return [], len(texts)

    arr = np.asarray(vecs, dtype=np.float32)
    n = arr.shape[0]
    sim = arr @ arr.T
    np.fill_diagonal(sim, 0.0)

    from sklearn.cluster import AgglomerativeClustering

    # cosine distance = 1 - similarity; average linkage merges on mean distance.
    dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - threshold,
    )
    labels = model.fit_predict(dist)

    ideas = []
    by_label = {}
    idxs_by_label = {}
    for i, lbl in enumerate(labels):
        key = int(lbl)
        by_label.setdefault(key, []).append(texts[i])
        idxs_by_label.setdefault(key, []).append(i)
    for lbl, members in by_label.items():
        if len(members) < min_idea_size:
            continue
        # Medoid = the member embedding nearest the cluster centroid (max
        # dot product in cosine space), resolved to its text string.
        idxs = idxs_by_label[lbl]
        sub = arr[idxs]
        centroid = sub.mean(axis=0)
        medoid_text = texts[idxs[int(np.argmax(sub @ centroid))]]
        ideas.append((medoid_text, members))
    return ideas, 0


def store_ideas(conn, ideas, threshold):
    """Replace stored Ideas with the given ones. Returns stats dict."""
    conn.execute("TRUNCATE thought_idea")
    conn.execute("TRUNCATE ideas RESTART IDENTITY CASCADE")
    stats = {"ideas": 0, "singletons": 0, "members": 0}
    for medoid_text, members in ideas:
        if len(members) < 2:
            stats["singletons"] += 1
            continue  # don't store singletons
        row = conn.execute(
            "INSERT INTO ideas (size, threshold, central) VALUES (%s, %s, %s) RETURNING id",
            (len(members), threshold, medoid_text),
        ).fetchone()
        for text in members:
            cur = conn.execute(
                """
                INSERT INTO thought_idea (thought_id, idea_id)
                SELECT id, %s FROM thoughts WHERE text = %s
                ON CONFLICT DO NOTHING
                """,
                (row[0], text),
            )
            stats["members"] += cur.rowcount
        stats["ideas"] += 1
    return stats