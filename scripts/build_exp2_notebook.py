#!/usr/bin/env python3
"""Build notebooks/exp2_lr_x_vs_xc_colab.ipynb.

Experiment 2 — the "one thing" test: identical logistic regression on
  A. X -> Y                       (control, already scored in exp1)
  B. X + C -> Y                   (supplement: the idea/C bottleneck)

Single variable changed (adding C), same 60/40 doc-level split, same seed,
same LR hyperparameters. Committed inputs:
  results/exp1/samples_2858.csv  (sentence_id, doc_key, sentence, label)
  results/exp1/sample_ideas.csv  (sentence_id, idea_id, central, size)

C is encoded as a multi-hot indicator over the 8,706-idea space (MultiLabelBinarizer),
so a linear probe over C has analytically exact per-idea -> label attribution.
"""

import json

MD = "markdown"
CODE = "code"

cells = []

cells.append({
    "cell_type": MD,
    "metadata": {},
    "source": [
        "# Experiment 2 — LR: X → Y vs X + C → Y\n",
        "\n",
        "The **one-thing test**: a single, identical logistic regression, with the\n",
        "only change being the addition of the semantic-bottleneck features C\n",
        "(the Ideas supporting each sentence). Same 60/40 **document-level** split,\n",
        "same seed, same hyperparameters.\n",
        "\n",
        "- **A (control):** TF-IDF sentence features → label\n",
        "- **B (experiment):** TF-IDF sentence features **+ C multi-hot over the\n",
        "  8,706-idea space** → label\n",
        "\n",
        "C is label-conditioned by construction (QSBC Phase 1: the teacher is shown Y\n",
        "and produces the general reasoning principles). This makes B the **ceiling**\n",
        "for the pipeline: it measures whether the idea layer carries signal worth\n",
        "extracting (~+10pt is the target). The deployed gain = ceiling × Stage-A\n",
        "extraction quality.\n",
        "\n",
        "Metrics: **accuracy + macro-F1** on the held-out test set. Per-class report\n",
        "for B, plus the exact per-idea → label attribution weights (linear probe).\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "# 1. Setup + fetch committed inputs\n",
        "import os\n",
        "os.chdir('/content')\n",
        "if not os.path.exists('/content/qsbc'):\n",
        "    !git clone -q https://github.com/jpeckenpaugh/qsbc.git qsbc\n",
        "os.chdir('/content/qsbc')\n",
        "print('cwd:', os.getcwd())\n",
        "!pip install -q pandas scikit-learn\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "df = pd.read_csv('results/exp1/samples_2858.csv')\n",
        "df = df.drop_duplicates(subset=['sentence_id']).reset_index(drop=True)\n",
        "ideas = pd.read_csv('results/exp1/sample_ideas.csv')\n",
        "\n",
        "# sentence -> list of idea_ids (the supporting C set)\n",
        "grp = ideas.groupby('sentence_id')['idea_id'].apply(list).to_dict()\n",
        "df['ideas'] = df['sentence_id'].map(grp).fillna('').apply(\n",
        "    lambda x: x if isinstance(x, list) else [])\n",
        "\n",
        "print('samples:', len(df), '| with >=1 idea:', int((df['ideas'].str.len() > 0).sum()))\n",
        "print('idea space size:', ideas['idea_id'].nunique())\n",
        "print('ideas per sample (mean):', round(df['ideas'].str.len().mean(), 2))\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "SEED = 42\n",
        "docs = df['doc_key'].unique()\n",
        "train_docs, test_docs = train_test_split(docs, test_size=0.4,\n",
        "                                          random_state=SEED)\n",
        "train = df[df['doc_key'].isin(train_docs)].reset_index(drop=True)\n",
        "test  = df[df['doc_key'].isin(test_docs)].reset_index(drop=True)\n",
        "print('doc overlap:', len(set(train_docs) & set(test_docs)))\n",
        "print('train:', len(train), '| test:', len(test))\n",
        "print('train classes:', train['label'].nunique(), '| test classes:', test['label'].nunique())\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.preprocessing import MultiLabelBinarizer\n",
        "from scipy.sparse import hstack\n",
        "\n",
        "# X features: TF-IDF over sentences (identical to exp1)\n",
        "tf = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),\n",
        "                     stop_words='english')\n",
        "Xtr = tf.fit_transform(train['sentence'])\n",
        "Xte = tf.transform(test['sentence'])\n",
        "print('X dims:', Xtr.shape)\n",
        "\n",
        "# C features: multi-hot over the full idea space\n",
        "mlb = MultiLabelBinarizer()\n",
        "mlb.fit(list(train['ideas']) + list(test['ideas']))\n",
        "Ctr = mlb.transform(list(train['ideas']))\n",
        "Cte = mlb.transform(list(test['ideas']))\n",
        "print('C dims:', Ctr.shape)\n",
        "\n",
        "XCtr = hstack([Xtr, Ctr]).tocsr()\n",
        "XCte = hstack([Xte, Cte]).tocsr()\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
        "\n",
        "def run(Xtr_f, Xte_f, name):\n",
        "    lr = LogisticRegression(max_iter=1000, C=10, class_weight='balanced')\n",
        "    lr.fit(Xtr_f, train['label'])\n",
        "    p = lr.predict(Xte_f)\n",
        "    acc = accuracy_score(test['label'], p)\n",
        "    f1 = f1_score(test['label'], p, average='macro')\n",
        "    print(f'{name:12s} acc={acc:.4f}  macro-F1={f1:.4f}')\n",
        "    return lr, p, acc, f1\n",
        "\n",
        "lrA, pA, accA, f1A = run(Xtr, Xte, 'A: X only')\n",
        "lrB, pB, accB, f1B = run(XCtr, XCte, 'B: X + C')\n",
        "\n",
        "print('\\nDelta (B - A):  acc %+.4f   macro-F1 %+.4f' % (accB - accA, f1B - f1A))\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "import pandas as pd\n",
        "rows = [\n",
        "    ['A: X -> Y (control)', accA, f1A],\n",
        "    ['B: X + C -> Y (experiment)', accB, f1B],\n",
        "]\n",
        "print(pd.DataFrame(rows, columns=['Condition', 'Accuracy', 'Macro-F1']).to_string(index=False))\n",
        "print('\\nPer-class report for B (X + C -> Y):')\n",
        "print(classification_report(test['label'], pB, zero_division=0))\n"
    ],
})

cells.append({
    "cell_type": MD,
    "metadata": {},
    "source": [
        "## Attribution (the linear-probe payoff)\n",
        "\n",
        "Because the head is strictly linear, per-idea → label attribution is exact:\n",
        "`weight(idea, label)`. Below, the top contributing ideas per label from the\n",
        "B-model's idea block."
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "# Extract the C-block weights (idea dims are the last Ctr.shape[1] columns)\n",
        "W = lrB.coef_                                  # (12, n_features)\n",
        "WC = W[:, -Ctr.shape[1]:]                     # (12, n_ideas)\n",
        "classes = lrB.classes_\n",
        "id2label_idx = {mlb.classes_[j]: j for j in range(Ctr.shape[1])}\n",
        "central = ideas.drop_duplicates('idea_id').set_index('idea_id')['central'].to_dict()\n",
        "\n",
        "for ci, c in enumerate(classes):\n",
        "    top = np.argsort(WC[ci])[::-1][:3]\n",
        "    print(f'\\n[{c}]')\n",
        "    for j in top:\n",
        "        idea = mlb.classes_[j]\n",
        "        print(f'   w={WC[ci][j]:+.3f}  {central.get(idea, \"\")[:100]}')\n"
    ],
})

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"colab": {}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}

with open("notebooks/exp2_lr_x_vs_xc_colab.ipynb", "w") as fh:
    json.dump(notebook, fh, indent=1)
print("wrote notebooks/exp2_lr_x_vs_xc_colab.ipynb with", len(cells), "cells")