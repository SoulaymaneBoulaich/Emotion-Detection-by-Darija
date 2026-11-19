"""Fine-tune DarijaBERT for 6-label multi-label emotion classification.

Owner: Soulaymane Boulaich
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from utils import load_config, set_seed, get_device


def train_darija_bert(config):
    set_seed(config["training"]["seeds"][0])
    device = get_device()

    model_name = config["model"]["darija_bert"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(config["report"]["all_labels"]),
        problem_type="multi_label_classification",
    ).to(device)

    # TODO: tokenize dataset, create Trainer, train, save best model
    print(f"Training {model_name} on {device} ...")


if __name__ == "__main__":
    config = load_config()
    train_darija_bert(config)