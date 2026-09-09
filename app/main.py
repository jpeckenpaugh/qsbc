from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from psycopg.types.json import Jsonb

from app.db import connect
from app.extract import extract_json_array

app = FastAPI(title="C3PA Explorer", version="0.1.0")
pool = connect()

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def startup():
    pool.open()


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
            return {"error": "not found"}
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
        return {"error": "not found"}
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
        return {"error": "sentence is not labeled with that label"}, 400
    status, parsed = extract_json_array(body.raw_response)
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
    return {"id": row[0], "parse_status": status, "parsed": parsed}


def build_prompt_input(sentence_id: int, label_id: int | None):
    """Assemble the INPUT payload: Sentence, Label, and Document = all
    single-label sentences in the same document carrying the same label."""
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
        doc_sentences = conn.execute(
            """
            SELECT v.text
            FROM v_single_label_sentences v
            WHERE v.doc_id = (SELECT doc_id FROM sentences WHERE id = %s)
              AND v.label_id = %s
            ORDER BY v.sentence_id
            """,
            (sentence_id, label_id),
        ).fetchall()
        doc_key = sent[2]
    return {
        "sentence_id": sentence_id,
        "doc_key": doc_key,
        "label_id": label_id,
        "label": label_name,
        "payload": {
            "Sentence": sent[1],
            "Label": label_name,
            "Document": [r[0] for r in doc_sentences],
        },
    }


def render_prompt(payload: dict, min_n: int, max_n: int, generalize: bool):
    import json as _json

    lines = [
        f"Examine the supplied INPUT. Explain why the `Sentence` from the given "
        f"`Document` carries the given `Label`. Simulate a chain-of-thought: "
        f"{min_n}-{max_n} statements of reasoning. Respond with ONLY a JSON array of strings."
    ]
    if generalize:
        lines.append(
            "\nPhrase each statement as a general reasoning principle — one that would "
            "apply to any `Sentence` exhibiting the same features, not an observation "
            "about this specific `Sentence` and/or `Document`."
        )
    lines.append("\nINPUT:")
    lines.append("```json")
    lines.append(_json.dumps(payload["payload"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append(
        f"\nOUTPUT: a JSON array of {min_n}-{max_n} distinct, self-contained statements. "
        "Nothing else — no preamble, no markdown."
    )
    return "\n".join(lines)


@app.get("/api/prompt/random")
def get_random_prompt(
    label_id: int | None = Query(default=None),
    min_n: int = Query(default=2, ge=1, le=10),
    max_n: int = Query(default=5, ge=1, le=10),
    generalize: bool = Query(default=True),
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
        return {"error": "not found"}
    return get_prompt(row[0], label_id=row[1], min_n=min_n, max_n=max_n, generalize=generalize)


@app.get("/api/prompt/{sentence_id}")
def get_prompt(
    sentence_id: int,
    label_id: int | None = Query(default=None),
    min_n: int = Query(default=2, ge=1, le=10),
    max_n: int = Query(default=5, ge=1, le=10),
    generalize: bool = Query(default=True),
):
    info = build_prompt_input(sentence_id, label_id)
    if info is None:
        return {"error": "not found"}
    return {
        **info,
        "min": min_n,
        "max": max_n,
        "generalize": generalize,
        "prompt": render_prompt(info, min_n, max_n, generalize),
    }


# -------------------------------------------------------------- SPA -----

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index(request: Request):
    return FileResponse(STATIC_DIR / "index.html")