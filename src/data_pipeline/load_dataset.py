"""Load the SemEval-2025 Task 11 Track A dataset from Hugging Face.

Owner: Fares Ilham
"""

from datasets import load_dataset


def load_semeval_darija(config_name=None, split="train"):
    """Load the dataset. If config_name is None, defaults to the config
    containing Algerian + Moroccan Arabic; filter_ary() keeps only 'ary' ids."""
    dataset = load_dataset("vgaraujov/semeval-2025-task11-track-a", name=config_name, split=split)
    return dataset


if __name__ == "__main__":
    ds = load_semeval_darija(split="train")
    print(ds)
    print(ds.features)
    print(ds[0])