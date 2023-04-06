import os
import sys

from src.spamdetection.training import train_llms, train_baselines
from src.spamdetection.preprocessing import init_datasets

if __name__ == "__main__":

    # Download and process datasets
    if os.path.exists("data") == False:
        init_datasets()

    # Train baseline models
    train_baselines(
        list(range(10)),
        ["ling", "sms", "spamassassin", "enron"],
        [4, 8, 16, 32, 64, 128, 256, 0.8],
        "test",
    )

    # Train LLMs
    train_llms(
        list(range(5)),
        ["ling", "sms", "spamassassin", "enron"],
        [4, 8, 16, 32, 64, 128, 256, 0.8],
        "test",
    )
