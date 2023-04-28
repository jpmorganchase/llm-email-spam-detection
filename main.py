# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

from src.spamdetection.training import train_llms, train_baselines
from src.spamdetection.preprocessing import init_datasets

if __name__ == "__main__":

    # Download and process datasets
    if os.path.exists("data") == False:
        init_datasets()

    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("outputs/png").mkdir(parents=True, exist_ok=True)
    Path("outputs/csv").mkdir(parents=True, exist_ok=True)

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
