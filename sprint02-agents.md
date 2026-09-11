# Task: Model / Agent / Persona Layering + Views

> **STATUS: COMPLETED.** This is a historical task specification, not current
> state. The `models` + `agents` tables are populated, `reasonings.model` was
> migrated to `reasonings.agent_id` (commit `964efd2`), and the gemma4 corpus
> is now **1,152** reasonings on disk (`results/gemma4/*.json`, imported via
> `scripts/import_gemma4.py`). The canonical, current concept lives in
> [`docs/QSBC_CONCEPT.md`](docs/QSBC_CONCEPT.md).

## Goal

Today, the system stores a single free-text producer string on each reasoning
(`reasonings.model`). We want to replace that with a proper **three-layer**
concept so the user can see *who/what* produced every derived truth, and filter
by it:

1. **Model** — the base model (e.g. `opencode-go/deepseek-v4-flash`,
   `gemma4:26b-a4b-it-q4_K_M`). Static characteristics: family, size,
   quantization, source, checkpoint.
2. **Agent** — an instance of a Model, configured with a scope (what it does)
   and runtime settings (temperature, seed, variant). One Model can have many
   Agents.
3. **Persona** — the agent's Identity / Personality / Profile / Role, stored as
   a single JSON field (soft, heterogeneous, not normalized).

The relationship chain becomes:
`Reasoning → Agent → Model`, where an Agent also carries its Persona.

This is grounded in the existing agent definition at
`.opencode/agents/cot.md`, which already encodes all three layers in one file
(model, temperature/variant/permission, and a `description` that is the
persona).

## Why we're doing this

- **"Model" is overloaded and ambiguous** (we discussed: a schema is a model, a
  diagram is a model, an LLM is a model…). We want precise vocabulary.
- The system has two real producers today that must be distinguishable and
  swappable: **deepseek** (3,023 reasonings in the DB) and **gemma4** (480
  reasonings in local JSON, not yet imported).
- The UI currently doesn't expose the producer as a first-class concept. We want
  Models and Agents to be browsable views, and the derived truths (Reasonings,
  Thoughts, Ideas) to be **filterable by Agent**.
- Longer-term vision (not this sprint): from *any* entity instance, a user
  should see the "threads" stretching left and right — every connection it has.
  This sprint is the first concrete cut of that.

## Current state (what exists)

- `reasonings` table has a free-text `model` column holding the producer string.
- There is no `models` or `agents` table.
- Derived data is recomputable from `reasonings` via `scripts/resync.py`
  (Thoughts are derived from Reasonings; Ideas are clustered from Thoughts).
- Two producers exist today:
  - `opencode-go/deepseek-v4-flash` — 3,023 reasonings in DB, across 6 runs
    (`bench-01..05`, `test-batch-2x12`).
  - `gemma4:26b-a4b-it-q4_K_M` — 480 reasonings in `results/gemma4/*.json`
    (478 parsed ok, 2 unparseable), run `colab-gemma4-01`, not yet in the DB.

## Schema changes

1. **New `models` table**
   - `id`, `name`, `family`, `size`, `quantization`, `source`, `checkpoint`.
   - One row per distinct base model.
   - Seeded with: `opencode-go/deepseek-v4-flash` and
     `gemma4:26b-a4b-it-q4_K_M` (with whatever characteristics we can derive).

2. **New `agents` table**
   - `id`, `name` (e.g. `cot`), `model_id` (FK → models),
     `temperature`, `seed`, `variant`, `scope`,
     `persona jsonb` (holds identity/personality/profile/role).
   - **Agent = a Model instantiated with a scope/settings** — keyed on
     `(model_id, name, temperature, variant)`, NOT run. `run_id` stays a
     non-key attribute on `reasonings`, not on the Agent.
   - Today that yields **2 agents**: `cot` on deepseek-v4-flash and `cot` on
     gemma4. A run (`bench-04`, `colab-gemma4-01`) is an *execution* of an
     Agent, not the Agent itself.

3. **`reasonings`** — replace the free-text `model` column with an
   `agent_id` FK → agents. The producer becomes a join: Reasoning → Agent →
   Model.

4. **Migration/resync** — `scripts/resync.py` currently drops and regenerates
   the derived tables (Thoughts, Ideas) but must NOT touch source data. The
   models/agents tables are new and need seeding; existing reasonings need to
   be mapped to the correct agent. We need a clear, safe path to populate
   models/agents and re-point reasonings without destroying the 3,023 existing
   reasonings.

## Data import

- Import the 480 gemma4 reasonings from `results/gemma4/*.json` into the DB so
  both producers are live. Each becomes a reasoning owned by the gemma4 Agent
  (run `colab-gemma4-01`).

## UI changes

1. **Models view** — lists base models with their characteristics and
   production counts.
2. **Agents view** — lists agent instances (model + run + settings + persona),
   with their production counts (reasonings, thoughts, distinct ideas).
3. **Agent filter** on the derived views (Reasonings, Thoughts) so a user can
   isolate one agent's interpretation.
4. **Provenance in the chain modals** — a Thought/Idea's chain already shows
   the reasoning's model; it should now show the Agent (and its Model).

## What "done" looks like

- `models` and `agents` tables exist and are populated (deepseek + gemma4).
- `reasonings.agent_id` is set; the free-text `model` string is gone.
- The 480 gemma4 reasonings are imported and attributed to the gemma4 Agent.
- Models and Agents are browsable views with production counts.
- Reasonings/Thoughts can be filtered by Agent.
- Thoughts/Ideas are still regenerable from Reasonings (resync still works), and
  the new agent attribution survives regeneration.
- No source truth (Labels, Documents, Sentences, Sentence/Label pairs) changes.

## Explicitly out of scope (this sprint)

- The full "select any entity, see all threads left and right" bidirectional
  explorer (future milestone). The chain modals are the partial version.
- Versioning personas over time (a future option; this sprint uses one persona
  JSON per agent).
- Persona as a separate normalized table (we chose a single `jsonb` field).

## Decisions already made (confirmed)

- **Agent = a Model instantiated with a scope/settings** — NOT run-per-agent.
  Today that is 2 agents: `cot` on deepseek and `cot` on gemma4. `run_id` is a
  non-key attribute on `reasonings`, not the Agent identity.
- **Drop the `reasonings.model` column now**, but keep the POST contract
  backward-compatible: the API still accepts `model` + `run_id`, and the server
  **resolves/creates the Agent** from the model (and known scope/settings) and
  stores `agent_id`. `run_batch.py` and the Colab notebook keep sending `model`.
- **Migration order**: add `reasonings.agent_id` as **nullable** → backfill via
  `(model, run_id) → agent` lookup → then make it `NOT NULL` and drop the
  `model` column.
- **Persona** = single flexible `jsonb` field on `agents`, derived from the
  agent `description` (e.g. the `cot.md` "C3PA expert" text) as
  `{role, style, variant, description}` — not rigidly four fixed keys.
- **Model metadata**: gemma4 characteristics exist (notebook: family gemma4,
  ~25.8B, Q4_K_M, Ollama). deepseek has no metadata in the repo — allow NULLs
  for unknown `family`/`size`/`quantization`/`source`/`checkpoint` fields.
- **gemma4 import**: a script that re-runs `extract_json_array(raw_response)`
  to populate `parsed` (the JSON has no `parsed` field), dedups on the FULL
  `(sentence_id, label_id, run_id)` triple (119 gemma4 pairs overlap deepseek
  pairs but with a different run — must not skip them), and seeds the gemma4
  agent BEFORE importing. The 2 unparseable → `parsed NULL`, status
  `unparseable` (matches the CHECK).
- **Filters**: keep the existing `run_id` filter AND add an agent filter on
  Reasonings/Thoughts (different granularity). **Defer** an Ideas agent-filter.
- **Response shape**: read endpoints return a flat `agent` + `model` string on
  each reasoning (minimal UI churn), not deeply nested objects.
- Models = normalized (stable shape); Agents + Persona carry the variable/soft
  data.
- Models + Agents views are the first concrete UI cut; agent filtering on
  derived views is included.