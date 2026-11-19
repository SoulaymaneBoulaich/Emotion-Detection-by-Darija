"""Predict-the-majority baseline.

Owner: Tasnim Touil
"""

from utils import load_config


def majority_baseline(config):
    print("Running predict-the-majority baseline ...")
    # TODO: predict most frequent label(s) for every example


if __name__ == "__main__":
    config = load_config()
    majority_baseline(config)