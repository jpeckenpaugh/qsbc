#!/usr/bin/env python3
"""Build notebooks/exp1_lr_vs_bert_colab.ipynb.

Experiment 1: given the 2,858 idea-mapped samples (sentence, label), a clean
document-level 60/40 train/test split, benchmark TF-IDF + Logistic Regression
vs fine-tuned BERT. Runs standalone in Colab; no FastAPI/Postgres needed.

The sample set is committed at results/exp1/samples_2858.csv (sentence_id, doc_key,
sentence, label), so the notebook is deterministic and identical to what the
local DB holds.
"""

import json

MD = "markdown"
CODE = "code"

cells = []

cells.append({
    "cell_type": MD,
    "metadata": {},
    "source": [
        "# Experiment 1 — LR vs BERT on the 2,858 Idea-Mapped Samples\n",
        "\n",
        "Benchmark of the 2,858 QSBC idea-mapped samples (single-label, 12 CPRA\n",
        "classes, balanced 195\u2013264/class). A **document-level 60/40** train/test\n",
        "split (no leakage: all sentences of a policy doc stay on one side),\n",
        "then two classifiers:\n",
        "\n",
        "1. **TF-IDF + Logistic Regression** (sklearn) — fast baseline.\n",
        "2. **Fine-tuned BERT** (`bert-base-uncased`) — HuggingFace Trainer,\n",
        "   3 epochs, lr 2e-5.\n",
        "\n",
        "Metrics on the held-out test set: **accuracy + macro-F1**.\n",
        "Reference (full 37,284-corpus) targets: LR 0.7935 / 0.7161,\n",
        "BERT 0.8208 / 0.7529. This experiment checks subset fidelity: does the\n",
        "idea-mapped slice reproduce the full-corpus ranking?\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "# 1. Setup + fetch the committed sample set\n",
        "import os\n",
        "os.chdir('/content')\n",
        "if not os.path.exists('/content/qsbc'):\n",
        "    !git clone -q https://github.com/jpeckenpaugh/qsbc.git qsbc\n",
        "os.chdir('/content/qsbc')\n",
        "print('cwd:', os.getcwd())\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "!pip install -q pandas scikit-learn transformers datasets torch accelerate\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "df = pd.read_csv('results/exp1/samples_2858.csv')\n",
        "df = df.drop_duplicates(subset=['sentence_id']).reset_index(drop=True)\n",
        "print(df.shape)\n",
        "print(df['label'].value_counts())\n"
    ],
})

cells.append({
    "cell_type": MD,
    "metadata": {},
    "source": [
        "## 2. Document-level 60/40 split\n",
        "\n",
        "Split by `doc_key` (not by row) to prevent leakage between train and test."
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
        "train_df = df[df['doc_key'].isin(train_docs)].reset_index(drop=True)\n",
        "test_df  = df[df['doc_key'].isin(test_docs)].reset_index(drop=True)\n",
        "\n",
        "# Leakage check: doc overlap must be 0\n",
        "print('train docs:', len(train_docs), '| test docs:', len(test_docs))\n",
        "print('doc overlap:', len(set(train_docs) & set(test_docs)))\n",
        "print('train samples:', len(train_df), '| test samples:', len(test_df))\n",
        "print('train classes:', train_df['label'].nunique(),\n",
        "      '| test classes:', test_df['label'].nunique())\n",
        "print('\\nclass distribution (%):')\n",
        "dist = pd.DataFrame({\n",
        "    'train': train_df['label'].value_counts(normalize=True) * 100,\n",
        "    'test': test_df['label'].value_counts(normalize=True) * 100,\n",
        "}).fillna(0).round(2)\n",
        "print(dist)\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "y_train = train_df['label'].astype(str)\n",
        "y_test  = test_df['label'].astype(str)\n",
        "classes = sorted(set(y_train) | set(y_test))\n",
        "print('num classes:', len(classes))\n"
    ],
})

cells.append({
    "cell_type": MD,
    "metadata": {},
    "source": [
        "## 3. Baseline — TF-IDF + Logistic Regression"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
        "\n",
        "lr_pipe = make_pipeline(\n",
        "    TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),\n",
        "                    stop_words='english'),\n",
        "    LogisticRegression(max_iter=1000, C=10, class_weight='balanced'),\n",
        ")\n",
        "lr_pipe.fit(train_df['sentence'], y_train)\n",
        "pred_lr = lr_pipe.predict(test_df['sentence'])\n",
        "print('LR accuracy:', round(accuracy_score(y_test, pred_lr), 4))\n",
        "print('LR macro-F1:', round(f1_score(y_test, pred_lr, average='macro'), 4))\n"
    ],
})

cells.append({
    "cell_type": MD,
    "metadata": {},
    "source": [
        "## 4. Fine-tuned BERT\n",
        "\n",
        "`bert-base-uncased`, 3 epochs, lr 2e-5, batch 16. Uses the HF Trainer."
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "from datasets import Dataset\n",
        "from transformers import (\n",
        "    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,\n",
        ")\n",
        "\n",
        "id2label = {i: c for i, c in enumerate(classes)}\n",
        "label2id = {c: i for i, c in id2label.items()}\n",
        "\n",
        "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n",
        "\n",
        "def tokenize(batch):\n",
        "    return tokenizer(batch['sentence'], truncation=True, padding='max_length',\n",
        "                     max_length=128)\n",
        "\n",
        "train_ds = Dataset.from_pandas(\n",
        "    pd.DataFrame({'sentence': train_df['sentence'],\n",
        "                  'label': train_df['label'].map(label2id)}))\n",
        "test_ds = Dataset.from_pandas(\n",
        "    pd.DataFrame({'sentence': test_df['sentence'],\n",
        "                  'label': test_df['label'].map(label2id)}))\n",
        "\n",
        "train_ds = train_ds.map(tokenize, batched=True)\n",
        "test_ds  = test_ds.map(tokenize, batched=True)\n",
        "train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])\n",
        "test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])\n",
        "\n",
        "model = AutoModelForSequenceClassification.from_pretrained(\n",
        "    'bert-base-uncased', num_labels=len(classes), id2label=id2label,\n",
        "    label2id=label2id)\n",
        "\n",
        "args = TrainingArguments(\n",
        "    output_dir='/content/bert_out',\n",
        "    num_train_epochs=3,\n",
        "    learning_rate=2e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=32,\n",
        "    seed=SEED,\n",
        "    report_to=[],\n",
        "    save_strategy='no',\n",
        "    logging_steps=50,\n",
        ")\n",
        "trainer = Trainer(model=model, args=args,\n",
        "                  train_dataset=train_ds, eval_dataset=test_ds)\n",
        "trainer.train()\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "import torch\n",
        "\n",
        "out = trainer.predict(test_ds)\n",
        "pred_bert = [id2label[int(x)] for x in np.argmax(out.predictions, axis=1)]\n",
        "print('BERT accuracy:', round(accuracy_score(y_test, pred_bert), 4))\n",
        "print('BERT macro-F1:', round(f1_score(y_test, pred_bert, average='macro'), 4))\n"
    ],
})

cells.append({
    "cell_type": CODE,
    "metadata": {},
    "source": [
        "# 5. Comparison table + per-class detail\n",
        "rows = [\n",
        "    ['TF-IDF + Logistic Regression',\n",
        "     round(accuracy_score(y_test, pred_lr), 4),\n",
        "     round(f1_score(y_test, pred_lr, average='macro'), 4)],\n",
        "    ['Fine-tuned BERT',\n",
        "     round(accuracy_score(y_test, pred_bert), 4),\n",
        "     round(f1_score(y_test, pred_bert, average='macro'), 4)],\n",
        "]\n",
        "import pandas as pd\n",
        "summary = pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Macro-F1'])\n",
        "print(summary.to_string(index=False))\n",
        "print('\\nReference (full 37,284 corpus): LR 0.7935 / 0.7161, BERT 0.8208 / 0.7529')\n",
        "print('\\nPer-class (LR):')\n",
        "print(classification_report(y_test, pred_lr, zero_division=0))\n",
        "print('\\nPer-class (BERT):')\n",
        "print(classification_report(y_test, pred_bert, zero_division=0))\n"
    ],
})

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"colab": {}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}

with open("notebooks/exp1_lr_vs_bert_colab.ipynb", "w") as fh:
    json.dump(notebook, fh, indent=1)
print("wrote notebooks/exp1_lr_vs_bert_colab.ipynb with", len(cells), "cells")