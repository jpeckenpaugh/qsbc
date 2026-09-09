"""Shared, dependency-free logic for building the locked cot prompt payload.

Used by both the FastAPI app (app/main.py) and the standalone Colab notebook
(which must NOT import FastAPI/psycopg). Keeping the payload construction and
prompt rendering here guarantees identical inputs to the model across runtimes.
"""

import json


def centered_window(texts, target_index, doc_limit):
    """Return a centered window of texts around the target (target included).

    Only applied when doc_limit is set and the group is larger; the target's
    index is used to center, clamped at the group edges."""
    total = len(texts)
    if doc_limit is None or total <= doc_limit or target_index < 0 or target_index >= total:
        return texts
    half = doc_limit // 2
    start = max(0, min(target_index - half, total - doc_limit))
    return texts[start:start + doc_limit]


def build_payload(sentence_text, label_name, doc_texts, target_index, doc_limit=None):
    """Assemble the locked INPUT payload: Sentence, Label, and Document.

    doc_texts: every same-label single-label sentence in the document, in
    document order. target_index: position of the target sentence in that list.
    """
    return {
        "Sentence": sentence_text,
        "Label": label_name,
        "Document": centered_window(doc_texts, target_index, doc_limit),
    }


def render_prompt(payload, min_n=2, max_n=5, generalize=True):
    """Render the full prompt text: instructions + INPUT JSON + output contract."""
    lines = [
        f"Examine the supplied INPUT. Explain why the `Sentence` from the given "
        f"`Document` carries the given `Label`. Simulate a chain-of-thought: "
        f"{min_n}-{max_n} statements of reasoning. Respond with ONLY a JSON array "
        f"of strings."
    ]
    if generalize:
        lines.append(
            "\nPhrase each statement as a general reasoning principle — one that "
            "would apply to any `Sentence` exhibiting the same features, not an "
            "observation about this specific `Sentence` and/or `Document`."
        )
    lines.append("\nINPUT:")
    lines.append("```json")
    lines.append(json.dumps(payload, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append(
        f"\nOUTPUT: a JSON array of {min_n}-{max_n} distinct, self-contained "
        "statements. Nothing else — no preamble, no markdown."
    )
    return "\n".join(lines)