"""Per-emotion decision threshold tuning on dev split.

Owner: Soulaymane Boulaich
"""

from utils import load_config


def tune_thresholds(config):
    print("Tuning per-emotion thresholds ...")
    # TODO: sweep thresholds per label to maximize F1 on dev


if __name__ == "__main__":
    config = load_config()
    tune_thresholds(config)