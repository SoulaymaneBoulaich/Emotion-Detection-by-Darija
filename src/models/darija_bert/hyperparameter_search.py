"""Hyperparameter search on dev split (lr, batch size, epochs, max_length).

Owner: Soulaymane Boulaich
"""

from utils import load_config


def run_hp_search(config):
    print("Running hyperparameter search ...")
    # TODO: iterate over lr / batch_size / epochs / max_length,
    # evaluate macro-F1 on dev, log best config


if __name__ == "__main__":
    config = load_config()
    run_hp_search(config)