"""Keep only Moroccan Darija examples (ids starting with 'ary').

Owner: Fares Ilham
"""

from datasets import Dataset


def filter_ary(dataset: Dataset) -> Dataset:
    """Filter rows whose example id starts with 'ary' (Moroccan Arabic subset)."""
    return dataset.filter(lambda example: example.get("id", "").startswith("ary"))


if __name__ == "__main__":
    from load_dataset import load_semeval_darija

    ds = load_semeval_darija(split="train")
    ds_ary = filter_ary(ds)
    print(f"Total: {len(ds)} | Moroccan (ary): {len(ds_ary)}")