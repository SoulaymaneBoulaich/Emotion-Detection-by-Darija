"""Per-emotion precision, recall, F1; macro-F1; micro-F1; mean/std over seeds.

Owner: Tasnim Touil
"""

from utils import load_config


def compute_metrics(y_true, y_pred, labels):
    """Return per-emotion and aggregate metrics."""
    print("Computing metrics ...")
    # TODO: precision_recall_f1_support per label, macro_f1, micro_f1


if __name__ == "__main__":
    config = load_config()
    print(config["report"])