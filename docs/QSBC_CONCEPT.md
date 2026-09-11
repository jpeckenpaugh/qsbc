# QSBC — The Quantized Semantic Bottleneck Classifier

**Single source of truth for the concept.** This document captures the full
idea: the motivation, the architecture, the design decisions, the empirical
evidence, and the deployment target. It exists so the concept never has to be
re-derived again.

---

## 1. The Thesis, in one sentence

> **Use 2026 technology (frontier LLMs, GPUs, TPUs, RAM, storage) to *compile* a
> frontier model's reasoning into a static, sub-megabyte, CPU-only artifact —
> a system of inference so small it could run on a computer from 1965.**

The entire QSBC idea is the attempt to make that sentence literally true.

### The deployment target behind the metaphor

"1965 hardware" is the *dramatization*. The actual engineering target is a
**low-profile / edge system**:

- No GPU, TPU, or NPU — only a modest CPU (or less)
- Small RAM (a few MB available for the model artifact)
- No online LLM dependency, no API calls, no connectivity required
- Power- and latency-constrained

The 1965 computer is the *constraint-enforcement device*: if the compiled
artifact can run on the hardest possible edge target, it runs anywhere a
low-profile system exists, with headroom to spare. It forces the honest
question for every component — *"can this be reduced to a static artifact that
needs no deep-network machinery at inference?"*

### The crisp analogy (from industry)

[Taalas](https://taalas.com/) builds "Hardcore Models": **"The Model is The
Computer"** — turning AI models into custom silicon rather than simulating them
as software on general-purpose hardware. Their thesis: a trained model's
weights are a *fixed program*; you should not pay general-compute overhead to
run it.

QSBC is the **software-only completion** of that same principle. Taalas keeps
the deep network (just hardwires it); QSBC *eliminates the deep-network runtime
entirely* by pre-compiling its logic into discrete predicates. The result
composes with Taalas's approach: QSBC reduces the model to the point where "the
model is the computer" is trivially, cheaply true — a few KB of lookup tables
whose "silicon" is essentially free — while remaining human-readable and
exactly attributable.

---

## 2. The Problem

### The task

Take sentences from real privacy policies and classify each one under one of
the **12 CPRA categories** (e.g., *Description of Right to Delete*, *Categories
of Personal Information Sold*). The reference dataset is the **C3PA** privacy
policy corpus.

**Reference baselines on the exact single-label 12-class split** (37,284
sentences / 399 documents):

| Model | Validation Accuracy | Macro-F1 |
| ----- | ------------------: | --------: |
| TF-IDF + Logistic Regression | 0.7935 | 0.7161 |
| Fine-tuned BERT (`bert-base-uncased`) | 0.8208 | 0.7529 |
| Fine-tuned FLAN-T5 (`flan-t5-base`) | — | — |

### Why the naive solutions fail

**X→Y lookup table:** a fixed-set lookup achieves 100% — but that's *retrieval*,
not *classification*. It memorizes the corpus and has zero generalization. The
held-out evaluation (unseen sentences from unseen documents) breaks it
completely. The lookup must live at the **concept** layer, not the sentence
layer.

**An edge transformer:** a small transformer is still a deep network — it needs
the runtime machinery (attention, layers, activations, model weights) that a
low-profile target rules out. Shrinking BERT does not make it 1965-runnable;
it makes it a smaller deep network.

---

## 3. The Architecture

### 3.1 The central equation

$$X \xrightarrow{\text{concept extraction}} \mathcal{C} \xrightarrow{\text{linear probe}} Y$$

- **X** — the input sentence
- **C** — a sparse, human-meaningful *concept* representation (a "semantic
  tokenization")
- **Y** — one of 12 labels

Instead of tokenizing at the subword level, QSBC tokenizes at the *semantic
proposition* level. An input like
*"Applications shall preserve the manifest when copying the asset."* becomes
concept IDs like `[17, 88, 431, 902]` → `NORMATIVE_REQUIREMENT,
PRESERVATION_BEHAVIOR, ASSOCIATED_METADATA, COPY_OPERATION`.

Because the downstream head is **strictly linear**, the attribution of any
concept to any class logit is **analytically exact**:

$$\text{Attribution}_{jk} = W_{jk} \cdot s_k$$

### 3.2 The "show your work" metaphor

The cleanest way to hold the whole system in your head is a law-school test:

- **X** — the *questions*. Each one is a sentence from a real privacy policy.
- **Y** — the *multiple-choice answers*. Which of the 12 CPRA categories
  applies to this sentence?
- **C** — the *"show your work"*. The 3–5 relevant ideas / reasoning
  propositions that point from X to Y — the equivalent of a student writing
  out the reasoning that justifies choosing answer Y rather than the other 11.

In ordinary classification, the model just bubbles in Y. QSBC makes the
student *show its work*: it writes down C (the reasoning), and the final
answer Y is derived from C.

The trick of QSBC is what happens to that "show your work" across the whole
exam. When a frontier model writes out reasoning for all ~37,000 questions,
those explanations collapse into a **finite vocabulary of recurring ideas**
(the concept alphabet C = {c_1...c_K}). So each question's "show your work"
becomes just a short list of concept IDs:

```
X: "You have the right to request that we correct inaccurate personal
    information that we maintain about you."
C: [NORMATIVE] [RIGHT_TO_CORRECT] [DATA_SUBJECT_REQUEST]
Y: Description of Right to Correct Information
```

Now the enormous, expensive, 2026 teacher is only needed *once*, to build the
alphabet and write the reasoning. The deployed system is a much weaker machine
that (1) reads the question, (2) matches it against the finite idea list, and
(3) scores the multiple-choice answers with a simple points scoreboard — the
"show your work" has been pre-compiled into a lookup.

### 3.3 The four phases

```
[Phase 1: Discovery]        (X, Doc, Y) ---> Frontier LLM ---> Raw rationale propositions

[Phase 2: Induction]        Raw propositions ---> Embed ---> Cluster ---> LLM consolidation
                                                              --> Concept alphabet C = {c_1..c_K}
                                                              --> Concept-label entropy audit

[Phase 3: Extraction]       Inference sentence (X) ---> Extractor ---> concept score vector s

[Phase 4: Decision]         s ---> quantization/gating ---> Linear LR head ---> Y + attribution map
```

**Phase 1 — Atomic proposition induction.** Conditioned on the sentence, its
source document, and the *true label*, a frontier LLM decomposes the reasoning
into 2–4 atomic, generalized statements ("this sentence mandates metadata
preservation").

**Phase 2 — Alphabet induction & consolidation.** Embed all propositions,
cluster them (HDBSCAN), and have a teacher LLM consolidate each cluster into a
single *canonical proposition* C_k. **Cluster ≠ concept** — clusters capture
topical proximity; consolidation produces identical *logical* propositions.
Audit with concept-label entropy H(Y|C_k): near-zero entropy means the concept
is a disguised label (reject); mid entropy means the concept composes across
labels (ideal).

**Phase 3 — Extraction.** Map a new sentence to a concept score vector s.
Two mechanisms, tested separately:
- **C_retrieval** — cosine similarity to concept medoids (cheap, zero-shot
  baseline)
- **C_student** — a small model *trained* to predict concepts from X (the core
  hypothesis)

**Phase 4 — Decision.** Gate/quantize s, feed a multinomial logistic regression
head, emit the label plus an exact per-concept attribution map.

### 3.4 Confidence, quantization, and gating

Moving from binary activations (c_k ∈ {0,1}) to continuous confidence
(s_k ∈ [0,1]) makes the bottleneck a *soft* concept bottleneck — the confidence
acts as a continuous gain on the weight W_jk.

Because downstream is strictly linear, fine-grained floats are both a noise
risk (background drift from hundreds of near-zero activations) and an
"epistemic precision illusion" (0.7184 implies certainty the model does not
possess). Options:

| Gating | Formula | Trade-off |
| ------ | ------- | --------- |
| Hard threshold | s'_k = s_k if s_k ≥ τ else 0 | kills noise; cliff edge at τ |
| Soft (shifted ReLU) | s'_k = max(0, s_k − τ) | continuous, keeps margin above floor |
| Top-M sparsity | keep the M highest | prevents token explosion |
| **Decile quantization** | round to {0.5, 0.6, ..., 1.0} | coarse tiers; mental-math auditing |

**Design stance:** quantization/gating must be an *ablation*, not an assumed
requirement. Test binary vs decile vs continuous; if deciles ≈ continuous, you
have demonstrated fine-grained concept confidence was unnecessary.

### 3.5 Information leakage (the soft-CBM risk)

A high-capacity intermediate representation lets the downstream head decode a
hidden embedding channel instead of interpreting s_k as "probability of concept
k". Safeguards: strictly 1-D per-concept scalars, probability calibration
(Platt/isotonic), and the linear head constraint (linear regression cannot
non-linearly decode a hidden state).

---

## 4. Design Decisions

- **The label-conditioned teacher is a feature, not leakage.** Phase 1 shows
  the teacher the *true label* by design. This is the compiler: the teacher's
  judgment is the ground truth being distilled into the alphabet. This makes
  the golden-B (teacher concepts) an **oracle ceiling** — the value available
  to a *label-blind* extractor, not the leak it would be in a vanilla
  classifier benchmark.
- **Every thought is an Idea.** Consolidation (≥2 similar thoughts merging) is
  an optimization on top, not a gate for existence. No thought is dropped.
- **The compile-time/runtime split.** All "thinking" (the 2026 LLM) happens
  once, at build time. The deployed device only executes compiled weights.

---

## 5. The Experimental Ladder

The ablation ladder isolates *where* performance comes from:

| Head | Inputs | Question | Role |
| ---- | ------ | -------- | ---- |
| LR(X) | TF-IDF | How well does linear surface text do? | floor |
| LR(C_teacher) | oracle concepts | Upper bound of the alphabet? | ceiling |
| LR(X + C_teacher) | both | Do concepts subsume lexical features? | residual |
| LR(C_retrieval) | top-k cosine | Does cheap retrieval suffice? | weak extractor |
| LR(X + C_retrieval) | both | Retrieval adds orthogonal signal? | residual |
| LR(C_student) | distilled concept logits | Can a small model learn the teacher's semantics? | **core hypothesis** |
| LR(X + C_student) | both | Concepts add signal beyond words? | residual |
| BERT / FLAN-T5 (X) | end-to-end | Unconstrained performance? | competitive target |

### Evidence so far (2,858 idea-mapped samples, doc-level 60/40 split)

| Experiment | Condition | Accuracy | Macro-F1 |
| ---------- | --------- | -------- | -------- |
| exp1 | LR: X → Y | 0.6559 | 0.6581 |
| exp1 | BERT: X → Y | 0.6468 | 0.6394 |
| exp2 | LR: X + C_teacher → Y (8,706 ideas) | **0.7541** | **0.7555** |
| exp2b | Frozen-BERT probe: X + C → Y | 0.4054 | 0.3734 |
| exp3 | LR: X + C_teacher shared-only (1,408) → Y | 0.7468 | 0.7488 |
| exp4 | LR: Stage-A ridge extractor → B̂ → Y | 0.6135 | 0.6142 |

**What the evidence shows:**
1. The +9.8pt ceiling from teacher concepts is real (exp2).
2. The signal lives in the **1,408 shared concepts**, not the 8,706 singleton
   tail (exp3) — a 6× alphabet reduction at −0.7pt cost. This is the Phase-2
   consolidation argument, made concrete, and it shrinks the extractor's label
   space to a tractable 1,408.
3. A naive linear extractor (ridge over TF-IDF) **overfits** — 0.915 train vs
   0.098 test recall (exp4). One point in the design space, not a refutation of
   extraction; the honest reading: **the first linear extraction method tried
   was unable to exploit the bottleneck.**

### Success criteria (from the review)

- **Interpretability success:** F1_QSBC ≥ F1_BERT − 0.05 (trade-off boundary)
- **Strong success:** F1_QSBC > F1_BERT
- **Exceptional:** F1_QSBC > F1_BERT and F1_FLAN-T5 with a much smaller
  extractor

---

## 6. The Open Question (what remains)

The central open problem is **Phase 3: extraction** — the cheapest runtime
procedure that maps a *new* sentence to its concept vector, given the compiled
alphabet + corpus, and that **generalizes across documents**.

The ceiling is measured (+9.8pt / shared-alphabet +9.1pt). The floor is
measured (0.656). The question is what sits between them:

> **How much of the +9pt concept ceiling can a label-blind, cheap extractor
> recover — and what is the cheapest artifact that does it?**

Candidate mechanisms: learned student classifier (deberta-v3-small or smaller),
retrieval over the concept-indexed corpus, or rules compiled from concept
medoids. This is the thesis to test next, on the ladder's rungs
LR(C_student) and LR(X + C_student).

---

## 7. Glossary

- **Concept Bottleneck Model (CBM):** an architecture that predicts
  human-meaningful concepts first, then the target from those concepts.
- **Semantic alphabet / concept alphabet:** the finite set of canonical
  propositions C = {c_1...c_K} induced from teacher rationales.
- **Golden B:** the set of 3–5 concepts that support a given sentence→label
  assignment, from the label-conditioned teacher. The oracle.
- **C_teacher / C_retrieval / C_student:** teacher oracle concepts, embedding
  retrieval concepts, and learned-extractor concepts respectively.
- **1965 constraint:** the deployment discipline — every runtime component must
  reduce to static sparse arithmetic, fitting in a few MB, no deep-network
  machinery.
- **Knowledge / reasoning distillation:** a large teacher model generates
  reasoning that a small student model trains on. QSBC is a discrete,
  linear-probe variant of this.