# qsbc — C3PA parsed database + chain-of-thought reasoning experiments

A PostgreSQL database (Docker) containing a parsed version of the
[C3PA privacy-policy dataset](https://github.com/MaazBinMusa/C3PA_Dataset),
plus tooling for experiments that ask agents to generate chain-of-thought
"reasonings" for sentence → CPRA-label assignments.

## Quick start

```bash
docker compose up -d --build          # Postgres 18 + app container (FastAPI + opencode + bun/node)
docker compose exec app bash scripts/load.sh   # import data/ TSVs into Postgres
```

The app container (`c3pa-app`) runs Python 3.14, FastAPI/uvicorn, and the
opencode CLI in one image. Credentials for the `opencode-go` provider are
mounted from `~/.local/share/opencode/auth.json` (read-only). The web UI is at
`http://localhost:8000`.

## Web viewer (FastAPI + Bootstrap SPA)

FastAPI + Bootstrap SPA exposes the parsed data (Stats, Documents, Sentences,
Labels, Reasonings, Prompt tabs, with search + pagination). Runs in the app
container — no host uvicorn needed.

API:
- `GET /api/stats`, `/api/labels`, `/api/documents?subset=&limit=&offset=`
- `GET /api/sentences?doc_id=&label_id=&label_mode=single|multi&q=&limit=&offset=`
- `GET /api/sentences/{id}` (includes labels + reasonings)
- `GET /api/reasonings?limit=&offset=`
- `POST /api/reasonings` — ingest an agent response `{sentence_id, label_id, run_id, model, raw_response}`; robustly extracts a JSON array of strings into `parsed` (status `ok`/`unparseable`, raw always preserved)
- `GET /api/prompt/random?label_id=&min_n=&max_n=&generalize=` — ready-to-paste prompt for a random single-label sentence
- `GET /api/prompt/{sentence_id}?label_id=&min_n=&max_n=&generalize=` — same, for a specific sentence

## Schema

| Table            | Purpose                                                                    |
| ---------------- | -------------------------------------------------------------------------- |
| `documents`      | One row per annotation CSV: `subset` (DB/WS), `doc_num`, `source_file`, `url` (best-effort from `Crawl/`) |
| `labels`         | The 13 CPRA labels (12 categories + `Others`)                              |
| `sentences`      | Sentences split from annotated passages, deduplicated per document         |
| `sentence_labels`| Junction table: a sentence can carry multiple labels (from multiple rows)  |
| `reasonings`     | Raw agent responses + extracted statements: `(sentence_id, label_id, run_id, model, raw_response, parsed jsonb, parse_status)` |

`v_single_label_sentences` is a view of sentences carrying exactly one label,
excluding `Others` — the 37,284-sentence corpus used by the notebook's
classification experiments.

## Pipeline notes (parity with the notebook)

- Document-level split identity is preserved via `doc_id` (e.g. `DB_1`, `WS_100`).
- Passages are segmented with `(?<=[.!?])\s+(?=[A-Z0-9])|\n+`, keeping sentences
  with ≥ 4 words. Literal `\n` escapes in source CSVs are **not** expanded, to
  exactly reproduce the notebook's published counts (37,284 sentences / 399 docs).
- Duplicate `(sentence, label)` pairs from repeated annotation rows are collapsed.

## Chain-of-thought experiments

The prompt-input format is locked as JSON — `Sentence`, `Label`, and `Document`
(an array of every single-label sentence in the same document carrying the same
label, including the target). The `/api/prompt/*` endpoints assemble this
directly from the DB so an agent only has to receive and answer.

Agent responses are stored raw and parsed post-hoc; fences and preamble are
fine — `app/extract.py` recovers the JSON array (exact parse → fence strip →
balanced-bracket scan → regex), and unparseable rows are flagged, never dropped.

## Running batches

The `cot` agent (`.opencode/agents/cot.md`, temp 0.05, `"*": deny`) is invoked
via the in-container opencode binary. Parse the dataset on the host first
(`python3 scripts/parse_c3pa.py`), then run a batch:

```bash
docker compose exec app python scripts/run_batch.py --per-label 24   # 288 runs, 4 workers
```

`run_batch.py` samples per-label via `/api/prompt/random`, runs the agent,
POSTs raw responses, and reports parse_status. `--label <id>` restricts to one
label, `--batch <run_id>` names the run (default `batch-<ts>`), `--dry-run`
samples without running.

```sql
-- the "unique reasoning statements" analysis
SELECT jsonb_array_elements_text(parsed) AS statement, count(*)
FROM reasonings WHERE parse_status = 'ok'
GROUP BY 1 ORDER BY 2 DESC;
```