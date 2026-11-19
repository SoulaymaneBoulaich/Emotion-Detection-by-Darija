"""Error analysis: group misclassified examples by cause; Integrated Gradients
with Captum on a sample of errors.

Owner: Tasnim Touil
"""

from utils import load_config


def run_error_analysis(config):
    print("Running error analysis ...")
    # TODO: confusion matrix, error grouping (spelling, code-switching, sarcasm, ambiguous),
    # Captum attributions


if __name__ == "__main__":
    config = load_config()
    run_error_analysis(config)