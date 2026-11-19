"""Fine-tune AraBERT or XLM-RoBERTa with same protocol as DarijaBERT.

Owner: Fares Ilham
"""

from utils import load_config


def train_second_encoder(config):
    model_name = config["model"]["second_encoder"]
    print(f"Training {model_name} ...")
    # TODO: same training loop as DarijaBERT


if __name__ == "__main__":
    config = load_config()
    train_second_encoder(config)