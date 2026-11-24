"""
Fine-tune DarijaBERT for 6-label multi-label emotion classification.
Owner: Soulaymane Boulaich

Data source priority:
  1. data/processed/ (Fares Ilham's train/dev/test splits) — preferred
  2. Hugging Face vgaraujov/semeval-2025-task11-track-a filtered to 'ary' ids — fallback
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from utils import load_config, set_seed, get_device

LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class DarijaEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def load_processed_splits(processed_dir: str = "data/processed"):
    """Load splits saved by Fares' build_splits.py (datasets.save_to_disk format)."""
    from datasets import load_from_disk

    path = Path(processed_dir)
    if not path.exists():
        return None
    try:
        ds = load_from_disk(str(path))
        return ds
    except Exception:
        return None


def load_hf_darija_fallback():
    """Load HF dataset and keep only Moroccan examples (ids starting with 'ary')."""
    from datasets import load_dataset

    ds = load_dataset("vgaraujov/semeval-2025-task11-track-a", split="train")
    ds = ds.filter(lambda ex: ex.get("id", "").startswith("ary"))
    return ds


def extract_labels(example, label_columns: Optional[list] = None):
    """Convert example to multi-hot label vector. Adjust keys to match your dataset schema."""
    if label_columns is None:
        label_columns = LABELS
    vec = []
    for l in label_columns:
        val = example.get(l, 0)
        try:
            vec.append(float(val))
        except (TypeError, ValueError):
            vec.append(0.0)
    return vec


def prepare_splits(config):
    """Return train/dev/dev datasets as lists of texts and label vectors."""
    ds_processed = load_processed_splits()
    if ds_processed is not None:
        # Fares' splits expected to have 'text' and label columns
        splits = {}
        for split_name in ["train", "dev", "test"]:
            if split_name in ds_processed and ds_processed[split_name] is not None:
                d = ds_processed[split_name]
                texts = d["text"]
                labels = [extract_labels(ex) for ex in d]
                splits[split_name] = (texts, labels)
        return splits

    # Fallback: HF direct load
    ds = load_hf_darija_fallback()
    # HF dataset may already be split; adjust if needed
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_texts = ds["train"]["text"]
    dev_texts = ds["test"]["text"]
    train_labels = [extract_labels(ex) for ex in ds["train"]]
    dev_labels = [extract_labels(ex) for ex in ds["test"]]
    return {
        "train": (train_texts, train_labels),
        "dev": (dev_texts, dev_labels),
        "test": (dev_texts, dev_labels),  # placeholder; replace with official test if available
    }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)
    preds = (predictions > 0.5).astype(int)

    # Per-label metrics
    results = {}
    for i, label in enumerate(LABELS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[f"{label}_precision"] = precision
        results[f"{label}_recall"] = recall
        results[f"{label}_f1"] = f1

    # Macro-F1
    f1s = [results[f"{l}_f1"] for l in LABELS]
    results["macro_f1"] = float(np.mean(f1s))
    # Micro-F1
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    results["micro_f1"] = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    return results


def train_darija_bert(config):
    set_seed(config["training"]["seeds"][0])
    device = get_device()

    model_name = config["model"]["darija_bert"]
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    splits = prepare_splits(config)
    train_texts, train_labels = splits["train"]
    dev_texts, dev_labels = splits["dev"]

    max_length = config["training"].get("max_length", 128)
    batch_size = config["training"].get("batch_size", 16)
    epochs = config["training"].get("epochs", 3)
    lr = config["training"].get("learning_rate", 2e-5)

    train_dataset = DarijaEmotionDataset(train_texts, train_labels, tokenizer, max_length)
    dev_dataset = DarijaEmotionDataset(dev_texts, dev_labels, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir="models/checkpoints/darija_bert",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="models/checkpoints/darija_bert/logs",
        logging_steps=10,
        report_to="none",
        seed=config["training"]["seeds"][0],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training ...")
    trainer.train()

    print("Evaluating on dev ...")
    dev_metrics = trainer.evaluate()
    print(f"Dev metrics: {json.dumps(dev_metrics, indent=2)}")

    # Save best model
    model_dir = Path("models/checkpoints/darija_bert/best_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Best model saved to {model_dir}")

    return trainer, dev_metrics


if __name__ == "__main__":
    config = load_config()
    train_darija_bert(config)