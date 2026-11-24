"""
Per-emotion decision threshold tuning on dev split.
Owner: Soulaymane Boulaich
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

from train import DarijaEmotionDataset, LABELS, load_config, get_device, prepare_splits


def get_dev_predictions(model, tokenizer, dev_texts, dev_labels, batch_size=16, max_length=128):
    device = get_device()
    dataset = DarijaEmotionDataset(dev_texts, dev_labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    return probs, labels


def tune_thresholds_for_label(y_true, y_probs, label_name):
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.95, 0.05):
        preds = (y_probs >= threshold).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def tune_thresholds(config):
    set_seed(42)
    device = get_device()
    model_name = config["model"]["darija_bert"]

    # Load best model if exists, else base model
    best_model_dir = Path("models/checkpoints/darija_bert/best_model")
    if best_model_dir.exists():
        model_path = str(best_model_dir)
        print(f"Loading best model from {model_path}")
    else:
        model_path = model_name
        print(f"Loading base model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    splits = prepare_splits(config)
    dev_texts, dev_labels = splits["dev"]

    max_length = config["training"].get("max_length", 128)
    batch_size = config["training"].get("batch_size", 16)

    probs, labels = get_dev_predictions(
        model, tokenizer, dev_texts, dev_labels, batch_size=batch_size, max_length=max_length
    )

    thresholds = {}
    for i, label in enumerate(LABELS):
        threshold, f1 = tune_thresholds_for_label(labels[:, i], probs[:, i], label)
        thresholds[label] = {
            "threshold": threshold,
            "f1_at_threshold": float(f1),
        }
        print(f"{label}: threshold={threshold:.2f}, F1={f1:.4f}")

    # Compute macro-F1 at tuned thresholds
    tuned_preds = np.zeros_like(probs)
    for i, label in enumerate(LABELS):
        tuned_preds[:, i] = (probs[:, i] >= thresholds[label]["threshold"]).astype(int)

    macro_f1s = []
    for i in range(len(LABELS)):
        y_true = labels[:, i]
        y_pred = tuned_preds[:, i]
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        macro_f1s.append(f1)

    print(f"\nMacro-F1 at tuned thresholds: {float(np.mean(macro_f1s)):.4f}")

    out_dir = Path("models/checkpoints/darija_bert")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to {out_dir / 'thresholds.json'}")

    return thresholds


if __name__ == "__main__":
    config = load_config()
    tune_thresholds(config)