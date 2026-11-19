"""Small BiLSTM neural baseline trained from scratch (PyTorch).

Owner: Tasnim Touil
"""

from utils import load_config


def train_bilstm(config):
    print("Training BiLSTM baseline ...")
    # TODO: embedding layer, BiLSTM, sigmoid head, BCE loss, 3 seeds


if __name__ == "__main__":
    config = load_config()
    train_bilstm(config)