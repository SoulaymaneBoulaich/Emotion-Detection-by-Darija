"""Build final train/dev/test splits and raw vs. normalized versions.

Owner: Fares Ilham
"""

import os
from datasets import DatasetDict

from load_dataset import load_semeval_darija
from filter_ary import filter_ary
from normalize import normalize_darija
from utils import ensure_dir


def build_splits(output_dir="data/processed", normalize_text=False):
    ensure_dir(output_dir)

    dataset = load_semeval_darija(split="train")
    dataset = filter_ary(dataset)

    # Split into train/dev (official test is separate on HF)
    dataset = dataset.train_test_split(test_size=0.15, seed=42)
    splits = DatasetDict({
        "train": dataset["train"],
        "dev": dataset["test"],
    })

    # Load official test if available
    try:
        test_ds = load_semeval_darija(split="test")
        test_ds = filter_ary(test_ds)
        splits["test"] = test_ds
    except Exception:
        splits["test"] = None

    # Apply normalization if requested
    if normalize_text:
        for split_name in splits:
            if splits[split_name] is not None:
                splits[split_name] = splits[split_name].map(
                    lambda ex: {"text": normalize_darija(ex["text"])}
                )

    # Save
    splits.save_to_disk(output_dir)
    print(f"Splits saved to {output_dir}")
    for k, v in splits.items():
        print(f"  {k}: {len(v) if v is not None else 'N/A'}")

    return splits


if __name__ == "__main__":
    build_splits(normalize_text=False)