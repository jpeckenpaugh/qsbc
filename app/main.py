from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from psycopg.types.json import Jsonb
from psycopg.errors import UniqueViolation

from app.db import connect
from app.extract import extract_json_array
from app.promptlib import build_payload, render_prompt
from app import thoughts, ideas
from app import embeddings

app = FastAPI(title="C3PA Explorer", version="0.1.0")
pool = connect()

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def startup():
    pool.open()
    with pool.connection() as conn:
        thoughts.ensure_schema(conn)
        ideas.ensure_schema(conn)


@app.on_event("shutdown")
def shutdown():
    pool.close()


# ---------------------------------------------------------------- API -----

@app.get("/api/stats")
def stats():
    with pool.connection() as conn:
        totals = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM documents),
              (SELECT count(*) FROM sentences),
              (SELECT count(*) FROM sentence_labels),
              (SELECT count(*) FROM labels),
              (SELECT count(*) FROM reasonings),
              (SELECT count(*) FROM v_single_label_sentences)
            """
        ).fetchone()
        per_label = conn.execute(
            """
            SELECT l.name, count(sl.sentence_id) AS n
            FROM labels l
            LEFT JOIN sentence_labels sl ON sl.label_id = l.id
            GROUP BY l.id, l.name
            ORDER BY n DESC
            """
        ).fetchall()
    return {
        "documents": totals[0],
        "sentences": totals[1],
        "sentence_labels": totals[2],
        "labels": totals[3],
        "reasonings": totals[4],
        "single_label_sentences": totals[5],
        "per_label": [{"label": r[0], "count": r[1]} for r in per_label],
    }


@app.get("/api/labels")
def labels():
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT l.id, l.name, count(sl.sentence_id) AS n
            FROM labels l
            LEFT JOIN sentence_labels sl ON sl.label_id = l.id
            GROUP BY l.id, l.name
            ORDER BY l.name
            """
        ).fetchall()
    return [{"id": r[0], "name": r[1], "count": r[2]} for r in rows]


@app.get("/api/documents")
def documents(
    subset: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    where, params = "", []
    if subset:
        where = "WHERE d.subset = %s"
        params.append(subset)
    with pool.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT d.id, d.subset, d.doc_num, d.source_file, d.url,
                   count(DISTINCT s.id) AS sentences,
                   count(DISTINCT sl.label_id) AS label_count
            FROM documents d
            LEFT JOIN sentences s ON s.doc_id = d.id
            LEFT JOIN sentence_labels sl ON sl.sentence_id = s.id
            {where}
            GROUP BY d.id
            ORDER BY d.subset, d.doc_num
            LIMIT %s OFFSET %s
            """,
            (*params, limit, offset),
        ).fetchall()
        total = conn.execute(
            f"SELECT count(*) FROM documents d {where}", params
        ).fetchone()[0]
    return {
        "total": total,
        "items": [
            {
                "id": r[0], "subset": r[1], "doc_num": r[2],
                "source_file": r[3], "url": r[4],
                "sentences": r[5], "label_count": r[6],
            }
            for r in rows
        ],
    }


@app.get("/api/sentences")
def sentences(
    doc_id: int | None = Query(default=None),
    label_id: int | None = Query(default=None),
    label_mode: str | None = Query(default=None, pattern="^(single|multi)$"),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    where, params = ["TRUE"], []
    if doc_id is not None:
        where.append("s.doc_id = %s")
        params.append(doc_id)
    if label_id is not None:
        where.append("EXISTS (SELECT 1 FROM sentence_labels sl WHERE sl.sentence_id = s.id AND sl.label_id = %s)")
        params.append(label_id)
    if label_mode == "single":
        where.append("(SELECT count(*) FROM sentence_labels sl WHERE sl.sentence_id = s.id) = 1")
    elif label_mode == "multi":
        where.append("(SELECT count(*) FROM sentence_labels sl WHERE sl.sentence_id = s.id) > 1")
    if q:
        where.append("s.text ILIKE %s")
        params.append(f"%{q}%")
    cond = " AND ".join(where)
    with pool.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT s.id, s.doc_id, d.doc_key, s.text,
                   array_agg(l.name ORDER BY l.name) AS labels
            FROM sentences s
            JOIN documents d ON d.id = s.doc_id
            LEFT JOIN sentence_labels sl ON sl.sentence_id = s.id
            LEFT JOIN labels l ON l.id = sl.label_id
            WHERE {cond}
            GROUP BY s.id, d.doc_key
            ORDER BY s.id
            LIMIT %s OFFSET %s
            """,
            (*params, limit, offset),
        ).fetchall()
        total = conn.execute(
            f"""
            SELECT count(*)
            FROM sentences s
            WHERE {cond}
            """,
            params,
        ).fetchone()[0]
    return {
        "total": total,
        "items": [
            {"id": r[0], "doc_id": r[1], "doc_key": r[2], "text": r[3], "labels": r[4]}
            for r in rows
        ],
    }


@app.get("/api/sentences/{sentence_id}")
def sentence_detail(sentence_id: int):
    with pool.connection() as conn:
        sent = conn.execute(
            """
            SELECT s.id, s.text, s.doc_id, d.subset, d.doc_num, d.doc_key, d.url
            FROM sentences s JOIN documents d ON d.id = s.doc_id
            WHERE s.id = %s
            """,
            (sentence_id,),
        ).fetchone()
        if sent is None:
            raise HTTPException(status_code=404, detail="not found")
        labels = conn.execute(
            """
            SELECT l.id, l.name FROM sentence_labels sl
            JOIN labels l ON l.id = sl.label_id
            WHERE sl.sentence_id = %s ORDER BY l.name
            """,
            (sentence_id,),
        ).fetchall()
        reasonings = conn.execute(
            """
            SELECT r.id, r.run_id, r.model, r.parse_status, r.parsed, r.raw_response, r.created_at
            FROM reasonings r WHERE r.sentence_id = %s ORDER BY r.created_at DESC
            """,
            (sentence_id,),
        ).fetchall()
    return {
        "id": sent[0], "text": sent[1], "doc_id": sent[2],
        "subset": sent[3], "doc_num": sent[4], "doc_key": sent[5], "url": sent[6],
        "labels": [{"id": r[0], "name": r[1]} for r in labels],
        "reasonings": [
            {
                "id": r[0], "run_id": r[1], "model": r[2],
                "parse_status": r[3], "parsed": r[4], "raw_response": r[5],
                "created_at": r[6].isoformat(),
            }
            for r in reasonings
        ],
    }


@app.get("/api/reasonings")
def reasonings(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.sentence_id, d.doc_key, s.text, l.name,
                   r.run_id, r.model, r.parse_status, r.parsed, r.created_at
            FROM reasonings r
            JOIN sentences s ON s.id = r.sentence_id
            JOIN documents d ON d.id = s.doc_id
            JOIN labels l ON l.id = r.label_id
            ORDER BY r.created_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()
        total = conn.execute("SELECT count(*) FROM reasonings").fetchone()[0]
    return {
        "total": total,
        "items": [
            {
                "id": r[0], "sentence_id": r[1], "doc_key": r[2],
                "text": r[3], "label": r[4], "run_id": r[5],
                "model": r[6], "parse_status": r[7], "parsed": r[8],
                "created_at": r[9].isoformat(),
            }
            for r in rows
        ],
    }


@app.get("/api/reasonings/{reasoning_id}")
def reasoning_detail(reasoning_id: int):
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT r.id, r.sentence_id, d.doc_key, s.text, l.name,
                   r.run_id, r.model, r.parse_status, r.parsed, r.raw_response, r.created_at
            FROM reasonings r
            JOIN sentences s ON s.id = r.sentence_id
            JOIN documents d ON d.id = s.doc_id
            JOIN labels l ON l.id = r.label_id
            WHERE r.id = %s
            """,
            (reasoning_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return {
        "id": row[0], "sentence_id": row[1], "doc_key": row[2],
        "text": row[3], "label": row[4], "run_id": row[5],
        "model": row[6], "parse_status": row[7], "parsed": row[8],
        "raw_response": row[9], "created_at": row[10].isoformat(),
    }


class ReasoningIn(BaseModel):
    sentence_id: int
    label_id: int
    run_id: str
    model: str | None = None
    raw_response: str


@app.post("/api/reasonings", status_code=201)
def create_reasoning(body: ReasoningIn):
    with pool.connection() as conn:
        valid = conn.execute(
            "SELECT 1 FROM sentence_labels WHERE sentence_id = %s AND label_id = %s",
            (body.sentence_id, body.label_id),
        ).fetchone()
    if valid is None:
        raise HTTPException(status_code=400, detail="sentence is not labeled with that label")
    status, parsed = extract_json_array(body.raw_response)
    try:
        with pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO reasonings
                    (sentence_id, label_id, run_id, model, raw_response, parsed, parse_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (body.sentence_id, body.label_id, body.run_id, body.model,
                 body.raw_response, Jsonb(parsed) if parsed is not None else None, status),
            ).fetchone()
            thoughts.insert_for_reasoning(
                conn, row[0], body.sentence_id, body.label_id, body.run_id, parsed
            )
    except UniqueViolation:
        raise HTTPException(status_code=409, detail="already exists")
    return {"id": row[0], "parse_status": status, "parsed": parsed}


def build_prompt_input(sentence_id: int, label_id: int | None, doc_limit: int | None = None):
    """Assemble the INPUT payload: Sentence, Label, and Document.

    Document = every single-label sentence in the same document carrying the
    same label, in document order. When doc_limit is set and the group is
    larger, a centered window around the target sentence is returned (target
    always included)."""
    with pool.connection() as conn:
        sent = conn.execute(
            """
            SELECT s.id, s.text, d.doc_key
            FROM sentences s JOIN documents d ON d.id = s.doc_id
            WHERE s.id = %s
            """,
            (sentence_id,),
        ).fetchone()
        if sent is None:
            return None
        if label_id is None:
            row = conn.execute(
                """
                SELECT sl.label_id, l.name
                FROM sentence_labels sl JOIN labels l ON l.id = sl.label_id
                WHERE sl.sentence_id = %s
                ORDER BY sl.label_id LIMIT 1
                """,
                (sentence_id,),
            ).fetchone()
            label_id, label_name = (row[0], row[1]) if row else (None, None)
        else:
            label_name = conn.execute(
                "SELECT name FROM labels WHERE id = %s", (label_id,)
            ).fetchone()
            label_name = label_name[0] if label_name else None
        if label_id is None or label_name is None:
            return None
        doc_rows = conn.execute(
            """
            SELECT v.sentence_id, v.text
            FROM v_single_label_sentences v
            WHERE v.doc_id = (SELECT doc_id FROM sentences WHERE id = %s)
              AND v.label_id = %s
            ORDER BY v.sentence_id
            """,
            (sentence_id, label_id),
        ).fetchall()
        doc_key = sent[2]
    doc_ids = [r[0] for r in doc_rows]
    doc_texts = [r[1] for r in doc_rows]
    target_index = doc_ids.index(sentence_id) if sentence_id in doc_ids else -1
    return {
        "sentence_id": sentence_id,
        "doc_key": doc_key,
        "label_id": label_id,
        "label": label_name,
        "doc_total": len(doc_texts),
        "payload": build_payload(sent[1], label_name, doc_texts, target_index, doc_limit),
    }


@app.get("/api/prompt/random")
def get_random_prompt(
    label_id: int | None = Query(default=None),
    min_n: int = Query(default=2, ge=1, le=10),
    max_n: int = Query(default=5, ge=1, le=10),
    generalize: bool = Query(default=True),
    doc_limit: int | None = Query(default=None, ge=1, le=100),
):
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT v.sentence_id, v.label_id
            FROM v_single_label_sentences v
            WHERE (%s::int IS NULL OR v.label_id = %s)
            ORDER BY random() LIMIT 1
            """,
            (label_id, label_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return get_prompt(row[0], label_id=row[1], min_n=min_n, max_n=max_n,
                      generalize=generalize, doc_limit=doc_limit)


@app.get("/api/prompt/{sentence_id}")
def get_prompt(
    sentence_id: int,
    label_id: int | None = Query(default=None),
    min_n: int = Query(default=2, ge=1, le=10),
    max_n: int = Query(default=5, ge=1, le=10),
    generalize: bool = Query(default=True),
    doc_limit: int | None = Query(default=None, ge=1, le=100),
):
    info = build_prompt_input(sentence_id, label_id, doc_limit=doc_limit)
    if info is None:
        return {"error": "not found"}
    return {
        **info,
        "min": min_n,
        "max": max_n,
        "generalize": generalize,
        "prompt": render_prompt(info["payload"], min_n, max_n, generalize),
    }


# -------------------------------------------------------------- Thoughts ----

@app.get("/api/thoughts/status")
def thoughts_status():
    with pool.connection() as conn:
        total = conn.execute("SELECT count(*) FROM reasonings").fetchone()[0]
        synced = conn.execute(
            "SELECT count(*) FROM reasonings r WHERE r.id <= %s",
            (thoughts.watermark(conn),),
        ).fetchone()[0]
        thoughts_total = conn.execute("SELECT count(*) FROM thoughts").fetchone()[0]
    return {
        "reasonings_total": total,
        "thoughts_synced": synced,
        "pending": max(0, total - synced),
        "thoughts_total": thoughts_total,
    }


@app.post("/api/thoughts/backfill")
def thoughts_backfill(
    batch_size: int = Query(default=500, ge=1, le=5000),
):
    with pool.connection() as conn:
        result = thoughts.backfill(conn, batch_size=batch_size)
    return result


@app.get("/api/thoughts")
def list_thoughts(
    label_id: int | None = Query(default=None),
    run_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    where, params = ["TRUE"], []
    if label_id is not None:
        where.append("c.label_id = %s")
        params.append(label_id)
    if run_id:
        where.append("c.run_id = %s")
        params.append(run_id)
    if q:
        where.append("c.norm LIKE %s")
        params.append(f"%{thoughts.normalize(q)}%")
    cond = " AND ".join(where)
    with pool.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.sentence_id, c.label_id, c.run_id, c.pos, c.text, c.norm,
                   d.doc_key, l.name, s.text
            FROM thoughts c
            JOIN reasonings r ON r.id = c.reasoning_id
            JOIN sentences s ON s.id = c.sentence_id
            JOIN documents d ON d.id = s.doc_id
            JOIN labels l ON l.id = c.label_id
            WHERE {cond}
            ORDER BY c.id
            LIMIT %s OFFSET %s
            """,
            (*params, limit, offset),
        ).fetchall()
        total = conn.execute(
            f"SELECT count(*) FROM thoughts c WHERE {cond}", params
        ).fetchone()[0]
    return {
        "total": total,
        "items": [
            {
                "id": r[0], "sentence_id": r[1], "label_id": r[2],
                "run_id": r[3], "pos": r[4], "text": r[5], "norm": r[6],
                "doc_key": r[7], "label": r[8], "sentence_text": r[9],
            }
            for r in rows
        ],
    }


@app.get("/api/thoughts/overlap")
def thoughts_overlap(
    label_id: int | None = Query(default=None),
    run_id: str | None = Query(default=None),
):
    where, params = ["TRUE"], []
    if label_id is not None:
        where.append("c.label_id = %s")
        params.append(label_id)
    if run_id:
        where.append("c.run_id = %s")
        params.append(run_id)
    cond = " AND ".join(where)
    with pool.connection() as conn:
        total_occ = conn.execute(
            f"SELECT count(*) FROM thoughts c WHERE {cond}", params
        ).fetchone()[0]
        # distinct norms that occur exactly once vs more than once
        uniq = conn.execute(
            f"""
            SELECT count(*) FROM (
                SELECT c.norm FROM thoughts c
                WHERE {cond} GROUP BY c.norm HAVING count(*) = 1
            ) u
            """,
            params,
        ).fetchone()[0]
        rep_occ = conn.execute(
            f"""
            SELECT count(*) FROM (
                SELECT c.norm FROM thoughts c
                WHERE {cond} GROUP BY c.norm HAVING count(*) > 1
            ) r
            """,
            params,
        ).fetchone()[0]
        rep_occ_count = conn.execute(
            f"""
            SELECT COALESCE(sum(cnt), 0) FROM (
                SELECT count(*) AS cnt FROM thoughts c
                WHERE {cond} GROUP BY c.norm HAVING count(*) > 1
            ) r
            """,
            params,
        ).fetchone()[0]
        max_occ = conn.execute(
            f"""
            SELECT COALESCE(max(cnt), 0) FROM (
                SELECT count(*) AS cnt FROM thoughts c
                WHERE {cond} GROUP BY c.norm
            ) g
            """,
            params,
        ).fetchone()[0]
        # distribution of occurrence counts
        dist = conn.execute(
            f"""
            SELECT cnt, count(*) AS norms FROM (
                SELECT count(*) AS cnt FROM thoughts c
                WHERE {cond} GROUP BY c.norm
            ) g GROUP BY cnt ORDER BY cnt
            """,
            params,
        ).fetchall()
    unique_occ = total_occ - rep_occ_count
    return {
        "total_occurrences": total_occ,
        "unique_thoughts": uniq,
        "repeated_thoughts": rep_occ,
        "unique_occurrences": unique_occ,
        "repeated_occurrences": rep_occ_count,
        "overlap_ratio": round(rep_occ_count / total_occ, 4) if total_occ else 0.0,
        "max_occurrences": max_occ,
        "distribution": [{"occurrences": r[0], "thoughts": r[1]} for r in dist],
    }


@app.get("/api/thoughts/aggregated")
def list_thoughts_aggregated(
    label_id: int | None = Query(default=None),
    run_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    min_occurrences: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    where, params = ["TRUE"], []
    if label_id is not None:
        where.append("c.label_id = %s")
        params.append(label_id)
    if run_id:
        where.append("c.run_id = %s")
        params.append(run_id)
    if q:
        where.append("c.norm LIKE %s")
        params.append(f"%{thoughts.normalize(q)}%")
    cond = " AND ".join(where)
    with pool.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT c.norm,
                   count(*)                       AS occurrences,
                   count(DISTINCT c.sentence_id)  AS sentences,
                   count(DISTINCT c.label_id)     AS labels,
                   count(DISTINCT c.run_id)       AS runs,
                   max(c.text)                    AS example
            FROM thoughts c
            WHERE {cond}
            GROUP BY c.norm
            HAVING count(*) >= %s
            ORDER BY occurrences DESC, c.norm
            LIMIT %s OFFSET %s
            """,
            (*params, min_occurrences, limit, offset),
        ).fetchall()
        total = conn.execute(
            f"""
            SELECT count(*) FROM (
                SELECT c.norm FROM thoughts c
                WHERE {cond}
                GROUP BY c.norm HAVING count(*) >= %s
            ) g
            """,
            (*params, min_occurrences),
        ).fetchone()[0]
    return {
        "total": total,
        "items": [
            {
                "norm": r[0], "occurrences": r[1], "sentences": r[2],
                "labels": r[3], "runs": r[4], "example": r[5],
            }
            for r in rows
        ],
    }


# -------------------------------------------------------------- Ideas ----

@app.get("/api/thoughts/ideas/status")
def idea_status():
    with pool.connection() as conn:
        ideas_count = conn.execute(
            "SELECT count(*) FROM ideas"
        ).fetchone()[0]
        last = conn.execute(
            "SELECT max(created_at) FROM ideas"
        ).fetchone()[0]
    return {
        "computed": ideas_count > 0,
        "ideas": ideas_count,
        "last": last.isoformat() if last else None,
        "model_available": embeddings.available(),
    }


@app.post("/api/thoughts/ideas")
def compute_ideas(
    threshold: float = Query(default=0.8, ge=0.0, le=1.0),
):
    """Recompute semantic Ideas over all distinct normalized thoughts."""
    with pool.connection() as conn:
        norms = [r[0] for r in conn.execute("SELECT DISTINCT norm FROM thoughts").fetchall()]
    if not norms:
        raise HTTPException(status_code=400, detail="no thoughts to cluster")
    ideas_list, failed = ideas.build_ideas(norms, threshold=threshold)
    with pool.connection() as conn:
        stats = ideas.store_ideas(conn, ideas_list, threshold)
    return {
        "norms": len(norms),
        "failed": failed,
        "threshold": threshold,
        **stats,
    }


@app.get("/api/thoughts/ideas")
def list_ideas(
    min_size: int = Query(default=2, ge=2),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT cl.id, cl.size, cl.threshold, cl.created_at,
                   array_agg(cnc.norm ORDER BY cnc.norm) AS norms
            FROM ideas cl
            JOIN thought_idea cnc ON cnc.idea_id = cl.id
            GROUP BY cl.id
            ORDER BY cl.size DESC, cl.id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()
        total = conn.execute("SELECT count(*) FROM ideas").fetchone()[0]
    return {
        "total": total,
        "items": [
            {
                "id": r[0], "size": r[1], "threshold": r[2],
                "created_at": r[3].isoformat(), "norms": r[4],
            }
            for r in rows
        ],
    }


# -------------------------------------------------------------- SPA -----

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index(request: Request):
    return FileResponse(STATIC_DIR / "index.html")