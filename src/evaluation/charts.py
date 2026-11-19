"""Charts: per-emotion F1 bars, training curves, label co-occurrence heatmap,
confusion matrix on single-emotion texts.

Owner: Tasnim Touil
"""

from utils import load_config


def plot_all(config):
    print("Generating charts ...")
    # TODO: matplotlib/seaborn figures saved to results/


if __name__ == "__main__":
    config = load_config()
    plot_all(config)