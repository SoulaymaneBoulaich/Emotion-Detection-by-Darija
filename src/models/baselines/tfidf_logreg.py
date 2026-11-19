"""TF-IDF character n-grams + one-vs-rest logistic regression baseline.

Owner: Tasnim Touil
"""

from utils import load_config


def train_tfidf_baseline(config):
    print("Training TF-IDF + Logistic Regression baseline ...")
    # TODO: char n-gram TF-IDF, OneVsRest(LogisticRegression), evaluate on dev/test


if __name__ == "__main__":
    config = load_config()
    train_tfidf_baseline(config)