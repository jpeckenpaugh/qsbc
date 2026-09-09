"""Cluster normalized claims by semantic similarity.

Approach: embed each distinct normalized claim with MiniLM, then group claims
whose pairwise cosine similarity exceeds a threshold into connected components
(single-linkage agglomerative). Run as an on-demand background job; results are
stored in `claim_clusters` + `claim_norm_cluster` so the UI can show buckets.
"""

import numpy as np

from app import embeddings

# ------------------------------------------------------------------ schema ----

CLUSTER_DDL = [
    """
    CREATE TABLE IF NOT EXISTS claim_clusters (
        id         BIGSERIAL PRIMARY KEY,
        size       INTEGER NOT NULL,
        threshold  REAL NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS claim_norm_cluster (
        norm       TEXT PRIMARY KEY,
        cluster_id BIGINT NOT NULL REFERENCES claim_clusters(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS claim_norm_cluster_cluster_idx ON claim_norm_cluster (cluster_id)",
]


def ensure_schema(conn):
    for ddl in CLUSTER_DDL:
        conn.execute(ddl)


# ---------------------------------------------------------------- clustering ----

def cluster_norms(norms, threshold=0.8, batch=1000, min_cluster_size=2):
    """Embed `norms` and cluster by average-linkage agglomerative clustering.

    Average linkage only merges groups whose *average* pairwise similarity is
    >= threshold, avoiding the single-linkage chaining that collapses claims
    sharing a common template opening.

    Returns (clusters, failed) where clusters is a list of list-of-norm-strings
    (size >= min_cluster_size) and failed is a count of unembeddable norms."""
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

    clusters = []
    by_label = {}
    for i, lbl in enumerate(labels):
        by_label.setdefault(int(lbl), []).append(norms[i])
    for members in by_label.values():
        if len(members) >= min_cluster_size:
            clusters.append(members)
    return clusters, 0


def store_clusters(conn, clusters, threshold):
    """Replace stored clusters with the given ones. Returns stats dict."""
    conn.execute("TRUNCATE claim_norm_cluster")
    conn.execute("TRUNCATE claim_clusters RESTART IDENTITY CASCADE")
    stats = {"clusters": 0, "singletons": 0, "members": 0}
    for c in clusters:
        if len(c) < 2:
            stats["singletons"] += 1
            continue  # don't store singletons
        row = conn.execute(
            "INSERT INTO claim_clusters (size, threshold) VALUES (%s, %s) RETURNING id",
            (len(c), threshold),
        ).fetchone()
        for norm in c:
            conn.execute(
                "INSERT INTO claim_norm_cluster (norm, cluster_id) VALUES (%s, %s)",
                (norm, row[0]),
            )
        stats["clusters"] += 1
        stats["members"] += len(c)
    return stats