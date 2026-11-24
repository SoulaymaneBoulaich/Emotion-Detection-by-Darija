"""
Hyperparameter search on dev split (lr, batch size, epochs, max_length).
Owner: Soulaymane Boulaich
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from train import prepare_splits, DarijaEmotionDataset, compute_metrics, LABELS
from utils import load_config, set_seed, get_device

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


def run_single_config(model, tokenizer, train_texts, train_labels, dev_texts, dev_labels, params):
    max_length = params["max_length"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    lr = params["learning_rate"]

    device = get_device()
    model = model.__class__.from_pretrained(
        params["model_name"],
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    train_dataset = DarijaEmotionDataset(train_texts, train_labels, tokenizer, max_length)
    dev_dataset = DarijaEmotionDataset(dev_texts, dev_labels, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=f"models/checkpoints/hp_search/{params['run_name']}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_dir=f"models/checkpoints/hp_search/{params['run_name']}/logs",
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    return metrics


def run_hp_search(config):
    set_seed(42)
    device = get_device()
    model_name = config["model"]["darija_bert"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splits = prepare_splits(config)
    train_texts, train_labels = splits["train"]
    dev_texts, dev_labels = splits["dev"]

    param_grid = {
        "learning_rate": config["training"].get("hp_learning_rates", [1e-5, 2e-5, 3e-5]),
        "batch_size": config["training"].get("hp_batch_sizes", [8, 16]),
        "epochs": config["training"].get("hp_epochs", [2, 3, 4]),
        "max_length": config["training"].get("hp_max_lengths", [64, 128]),
    }

    results = []
    best_macro_f1 = -1.0
    best_params = None

    keys = list(param_grid.keys())
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        params["model_name"] = model_name
        params["run_name"] = f"lr{params['learning_rate']}_bs{params['batch_size']}_ep{params['epochs']}_ml{params['max_length']}"

        print(f"Running config: {params['run_name']}")
        metrics = run_single_config(
            None, tokenizer, train_texts, train_labels, dev_texts, dev_labels, params
        )
        macro_f1 = metrics.get("eval_macro_f1", 0.0)
        params["macro_f1"] = macro_f1
        results.append(params)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_params = params

    print(f"\nBest macro-F1: {best_macro_f1:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    out_dir = Path("models/checkpoints/hp_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hp_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best_params}, f, indent=2)


if __name__ == "__main__":
    config = load_config()
    run_hp_search(config)