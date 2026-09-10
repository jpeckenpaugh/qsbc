"""Build Ideas by grouping similar normalized Thoughts by semantic similarity.

Approach: embed each distinct normalized thought with MiniLM, then group
thoughts whose pairwise cosine similarity exceeds a threshold into connected
components (agglomerative). Run as an on-demand background job; results are
stored in `ideas` + `thought_idea` so the UI can show buckets.
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
        central_norm TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS thought_idea (
        norm     TEXT PRIMARY KEY,
        idea_id  BIGINT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS thought_idea_idea_idx ON thought_idea (idea_id)",
]


def ensure_schema(conn):
    for ddl in SCHEMA_DDL:
        conn.execute(ddl)
    conn.execute("ALTER TABLE ideas ADD COLUMN IF NOT EXISTS central_norm TEXT")


# ---------------------------------------------------------------- clustering ----

def build_ideas(norms, threshold=0.8, batch=1000, min_idea_size=2):
    """Embed `norms` and group into Ideas by average-linkage agglomerative clustering.

    Average linkage only merges groups whose *average* pairwise similarity is
    >= threshold, avoiding the single-linkage chaining that collapses thoughts
    sharing a common template opening.

    Returns (ideas, failed) where ideas is a list of (medoid_norm, members)
    tuples (each cluster with size >= min_idea_size; members is a list of
    norm strings) and failed is a count of unembeddable norms."""
    if not norms:
        return [], 0
    vecs = embeddings.encode_batch(norms)
    if vecs is None:
        return [], len(norms)

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
        by_label.setdefault(key, []).append(norms[i])
        idxs_by_label.setdefault(key, []).append(i)
    for lbl, members in by_label.items():
        if len(members) < min_idea_size:
            continue
        # Medoid = the member embedding nearest the cluster centroid (max
        # dot product in cosine space). Uses row indices, not norms.
        idxs = idxs_by_label[lbl]
        sub = arr[idxs]
        centroid = sub.mean(axis=0)
        medoid_norm = norms[idxs[int(np.argmax(sub @ centroid))]]
        ideas.append((medoid_norm, members))
    return ideas, 0


def store_ideas(conn, ideas, threshold):
    """Replace stored Ideas with the given ones. Returns stats dict."""
    conn.execute("TRUNCATE thought_idea")
    conn.execute("TRUNCATE ideas RESTART IDENTITY CASCADE")
    stats = {"ideas": 0, "singletons": 0, "members": 0}
    for medoid_norm, members in ideas:
        if len(members) < 2:
            stats["singletons"] += 1
            continue  # don't store singletons
        row = conn.execute(
            "INSERT INTO ideas (size, threshold, central_norm) VALUES (%s, %s, %s) RETURNING id",
            (len(members), threshold, medoid_norm),
        ).fetchone()
        for norm in members:
            conn.execute(
                "INSERT INTO thought_idea (norm, idea_id) VALUES (%s, %s)",
                (norm, row[0]),
            )
        stats["ideas"] += 1
        stats["members"] += len(members)
    return stats