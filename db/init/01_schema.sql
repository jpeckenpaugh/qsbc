-- ============================================================
-- C3PA parsed database schema
-- Documents, Sentences, Labels, Reasonings
-- ============================================================

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    subset      TEXT NOT NULL CHECK (subset IN ('DB', 'WS')),
    doc_num     INTEGER NOT NULL,
    source_file TEXT NOT NULL UNIQUE,          -- e.g. 'DB/1.csv'
    doc_key     TEXT NOT NULL UNIQUE,          -- e.g. 'DB_1'
    url         TEXT,                          -- best-effort from Crawl/{db,ws}.csv
    UNIQUE (subset, doc_num)
);

CREATE TABLE IF NOT EXISTS labels (
    id   SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sentences (
    id      SERIAL PRIMARY KEY,
    doc_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text    TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS sentences_doc_id_md5_text_idx
    ON sentences (doc_id, md5(text));  -- hash-based: text can exceed btree row-size limit

CREATE INDEX IF NOT EXISTS sentences_doc_id_idx ON sentences (doc_id);

CREATE TABLE IF NOT EXISTS sentence_labels (
    sentence_id INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    label_id    INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    PRIMARY KEY (sentence_id, label_id)
);

CREATE TABLE IF NOT EXISTS reasonings (
    id           SERIAL PRIMARY KEY,
    sentence_id  INTEGER NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    label_id     INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    run_id       TEXT NOT NULL,                 -- experiment/agent-invocation id
    model        TEXT,                          -- agent/model that produced it
    raw_response TEXT NOT NULL,                 -- verbatim model output (fences, preamble included)
    parsed       JSONB,                         -- extracted JSON array of statements (NULL if unparseable)
    parse_status TEXT NOT NULL DEFAULT 'ok' CHECK (parse_status IN ('ok', 'unparseable')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (sentence_id, label_id, run_id)
);

CREATE INDEX IF NOT EXISTS reasonings_sentence_id_idx ON reasonings (sentence_id);

-- Convenience view: sentences that carry exactly one label (as used by the
-- notebook's classification experiments, excluding 'Others').
CREATE OR REPLACE VIEW v_single_label_sentences AS
SELECT s.id AS sentence_id,
       s.doc_id,
       s.text,
       l.id   AS label_id,
       l.name AS label
FROM sentences s
JOIN sentence_labels sl ON sl.sentence_id = s.id
JOIN labels l           ON l.id = sl.label_id
WHERE (SELECT count(*) FROM sentence_labels sl2 WHERE sl2.sentence_id = s.id) = 1
  AND l.name <> 'Others';