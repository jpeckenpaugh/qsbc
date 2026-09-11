# Sprint 01 — QSBC: The Bounded Expert

> **STATUS: COMPLETED.** This is a historical planning record. All proposed
> entity renames were executed (`claims`→`thoughts`, `claim_clusters`→`ideas`,
> `claim_norm_cluster`→`thought_idea`, `claims_sync`→`thoughts_sync`). The
> canonical, current concept lives in
> [`docs/QSBC_CONCEPT.md`](docs/QSBC_CONCEPT.md).

## Background & Framing

This project builds the **Quantized Semantic Bottleneck Classifier (QSBC)** — an
interpretable, auditable sentence-classification architecture applied to the
C3PA privacy-policy corpus. Rather than mapping text directly to labels
($X \to Y$) with an uninterpretable transformer, QSBC inserts a
human-meaningful **semantic proposition bottleneck** between input and
prediction:

$$X \xrightarrow{\text{concept extraction}} \mathcal{C} \xrightarrow{\text{linear probe}} Y$$

The reference baselines on the exact single-label 12-class C3PA split:

| Model | Validation Accuracy | Macro-F1 |
| ----- | ------------------: | --------: |
| TF-IDF + Logistic Regression | 0.7935 | 0.7161 |
| Fine-tuned BERT (`bert-base-uncased`) | 0.8208 | 0.7529 |

Sprint 01 is a **conceptual** sprint. It does not add code; it fixes the
*concept* of what the system is, why it is built this way, and the vocabulary
the UI and codebase should use. The output is a formalized entity hierarchy and
a clear statement of the architectural bet.

## The Core Concept

The goal is a **specialized expert model** that is an authority on a single,
closed topic (e.g. the CCPA), knowing nothing beyond that slice of the world.
The architecture compiles the entire knowledge of the domain down into a small
set of bounded lookup tables, so that *all* inference — understanding, decision,
and expression — is table lookup running on trivially simple hardware.

The key inversion: instead of keeping a large model running forever, we run a
frontier model **once** (at compile time), let it "think", and discard it. The
output of that one-time thought is distilled into bounded, finite codebooks.
At runtime the system does not think; it **remembers** — it was pre-judged.

The whole pipeline is a cascade of bounded lookups:

```
Table 1:  sentence  → active Ideas            (recognition / extraction)
Table 2:  Idea      → label votes             (decision)
Table 3:  Idea × language → token-string      (expression / rendering)
```

Every table is finite because the domain is closed. The "intelligence" has been
moved entirely into the compile-time construction; nothing is left to reason
about at execution time.

### The enabling premise: boundedness

The input is **bound by the knowledge space.** The contract is fixed and small:
*"Here is one sentence; apply one of 12 in-scope labels to it."* Anything outside
that space does not belong in the system. Out-of-domain input is not a hard case to
solve — it is out of scope *by definition*.

The "12 labels" is a deliberate reduction of C3PA's 13 categories: the "Others"
catch-all label (for "not one of the real categories") is excluded as out of
scope, and multi-label sentences are excluded so that **each input maps to
precisely one label**.

This is not a limitation to work around; it is the **enabling premise** of the
entire project. Because the input is bounded, the recognition problem is closed.
Because the label space is closed, the decision problem is closed. Because the
renderings are finite, the expression problem is closed. The system is closed on
every side — which is exactly why it can run on 1965 hardware at a fraction of
the inference cost of a generalist, and why it can never hallucinate beyond its
slice: **there is no "outside" inside it.**

### The information-theoretic framing

Human language is already the first compression of thought — a single idea costs
roughly 5–10 tokens in *any* of thousands of languages. An **Idea** is the
invariant that survives translation across all those codecs; it lives *below*
the language layer. Unpacking an Idea into a language is itself just another
bounded lookup (Table 3).

A frontier model, by contrast, "thinks" in a chain-of-thought that spends
hundreds of tokens per idea and re-derives the same ideas at full price on every
encounter. It pays continuously for probabilistic symbols that must be
re-interpreted. The QSBC pays once for referential symbols that are already
interpreted — meaning-by-reference is free.

Compression here is **not lossy in the meaningful sense**: the Idea is stored
as the *generator* of all its instantiations, and can always be unpacked back
into human-legible tokens. Nothing that matters is discarded.

### The two measurable claims + one property

1. **F1 above BERT.** The `LR(C)` probes must clear the black-box bar
   (0.8208 / 0.7529) on the single-label 12-class split. `LR(C_teacher)` is the
   ceiling check; `LR(C_student)` is the real target.
2. **Inference cost ~0.** Versus BERT's ~110M-parameter transformer: a couple of
   bounded lookups and a handful of additions.
3. **Unpackable.** Ideas render back to English (or any language) via a third
   bounded table, with minimal extra computation.

## The Bounded Input Domain (C3PA)

The knowledge space is not an abstraction — it is **countable**:

- **13** labels (12 in-scope; "Others" is the out-of-scope catch-all)
- **400** documents (source territory)
- **56,914** sentences (raw material)
- **81,000** sentence→label pairings (full relations)
- **40,505** single-label sentences; **16,409** multi-label sentences
- **37,284** single-label sentences excluding "Others" (the actual classification task)

The boundary is imposed by the dataset itself. The C3PA corpus *is* the domain,
closed and given. The chip is the embodiment of that finite, closed space.

The experiment's bounded world is reached by two exclusions from the raw
dataset: the **"Others"** catch-all label is dropped (13 → 12), and **multi-label
sentences** are dropped so that **each input sentence maps to precisely one
label** (40,505 → 37,284).

## Entity Hierarchy & Formalized Vocabulary

The UI hierarchy, top to bottom:

```
C3PA → Labels → Sentences → Documents → Reasonings → Thoughts → Ideas
```

| UI entity | Semantic definition | Current code/DB |
|---|---|---|
| C3PA | the dataset root; the given territory | — (not an entity) |
| Labels | the fixed decision categories | `labels` |
| Sentences | excerpts referencing the law | `sentences` |
| Documents | documents containing Sentences | `documents` |
| Reasonings | a collection of Thoughts for one `(sentence, label, run)` | `reasonings` |
| Thoughts | a single atomic statement / instantiation of an Idea | `claims` |
| Ideas | the aggregate of many similar Thoughts (the invariant) | `claim_clusters` |

### Semantic definitions

- **Reasoning** — a collection of Thoughts produced for one
  `(sentence, label, run)`. Stored raw + parsed.
- **Thought** — a single atomic proposition; one instance / permutation of an
  Idea.
- **Idea** — the consolidated invariant shared by many similar Thoughts. The
  bounded semantic atom everything reduces to.

Ideas are **not** "True" or "False". The operative axis is **Relevancy**: for a
given sentence and label, some Ideas are good notes leading to the correct
determination; others have nothing to do with the sentence. Relevance is
measurable (it is the probe coefficient); truth is not. This is the pivot from
correspondence (true/false) to instrumentality (relevant/irrelevant).

## Proposed Renames

To align the codebase and UI with the hierarchy above, rename the "claim"
vocabulary to "thought", and "cluster" to "idea":

| Current | Proposed |
|---|---|
| `claims` | `thoughts` |
| `claim_clusters` | `ideas` |
| `claim_norm_cluster` | `thought_idea` (many-to-one: thought → idea) |
| `claims_sync` | `thoughts_sync` |
| `claim` vocabulary in `claims.py` / `clusters.py` | `thought` |

Untouched: `reasonings`, `sentences`, `documents`, `labels`.

The `thought_idea` lookup — binding Thoughts to their consolidated Idea — is the
reduction that makes both the compilation and the inference architecture
bounded.

## Open Design Question

**Who enforces the boundary?** Two options, both compatible with boundedness:

1. **Gated upstream** — the system assumes in-domain input; something before it
   (a curated feed / gate) ensures only CCPA sentences arrive.
2. **Self-gating** — the chip carries a thin rejection test ("is this in the
   knowledge space?") and declines out-of-domain input.

The choice reflects how much we trust the world to bring only what belongs,
versus building the refusal into the chip.