import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots
import pandas as pd
import numpy as np
import random
import torch
import glob
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from src.spamdetection.preprocessing import get_dataset

plt.style.use("science")

SCORING = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "accuracy": accuracy_score,
}


def set_seed(seed) -> None:
    """Fix random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_loss(experiment: str, dataset_name: str, model_name: str) -> None:
    """Plot loss curve for LLMs."""
    log = pd.read_csv(f"outputs/csv/loss_{model_name}_{experiment}.csv")
    log = pd.DataFrame(log).iloc[:-1]

    train_losses = log["train_loss"].dropna().values
    eval_losses = log["eval_loss"].dropna().values
    x = np.arange(1, len(train_losses) + 1, step=1)

    with plt.style.context(["science", "high-vis"]):
        fig, ax = plt.subplots()
        plt.plot(x, train_losses, label="Training loss")
        plt.plot(x, eval_losses, label="Evaluation loss")

        ax.set_title(f"{model_name} ({dataset_name.upper()})")
        ax.set_xticks(x, labels=range(1, len(x) + 1))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")

        Path(f"outputs/pdf/").mkdir(parents=True, exist_ok=True)
        Path(f"outputs/png/").mkdir(parents=True, exist_ok=True)

        plt.savefig(f"outputs/pdf/loss_{model_name}_{experiment}.pdf", format="pdf")
        plt.savefig(
            f"outputs/png/loss_{model_name}_{experiment}.png", format="png", dpi=300
        )
        plt.show()


def plot_scores(experiment: str, dataset_name: str) -> None:
    """Plot scores as histogram."""
    scores = pd.read_csv(f"outputs/csv/{experiment}.csv", index_col=0)

    x = np.arange(len(scores))
    width = 0.2

    # Plot
    fig, ax = plt.subplots(figsize=(9, 3))
    rects1 = ax.bar(x=x - width, height=scores["f1"], width=width, label="F1 score")
    rects2 = ax.bar(x=x, height=scores["precision"], width=width, label="Precision")
    rects3 = ax.bar(x=x + width, height=scores["recall"], width=width, label="Recall")

    ax.set_title(f"{dataset_name.upper()}")
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels=scores.index, fontsize=10)
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc="lower center", ncol=4)

    fig.tight_layout()

    Path(f"outputs/pdf/").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/png/").mkdir(parents=True, exist_ok=True)

    plt.savefig(f"outputs/pdf/{experiment}.pdf", format="pdf")
    plt.savefig(f"outputs/png/{experiment}.png", format="png", dpi=300)
    plt.show()


def plot_pie_charts() -> None:
    """Plot ham/spam distribution for each dataset."""
    dataset_names = ["ling", "sms", "spamassassin", "enron"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    for i, dataset_name in enumerate(dataset_names):
        df = get_dataset(dataset_name)
        axs[i].pie(
            df["label"].value_counts().to_numpy(),
            autopct="%1.2f\%%",
            pctdistance=0.35,
            startangle=-30,
            wedgeprops={"width": 0.3},
            textprops={"fontsize": 22},
        )
        axs[i].set_title(f"{dataset_name.upper()}\n({len(df):,} samples)", fontsize=24)

    fig.legend(
        ["spam", "ham"],
        bbox_to_anchor=(0.5, -0.1),
        loc="lower center",
        ncol=2,
        prop={"size": 22},
    )

    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.3)
    plt.savefig(f"outputs/pdf/pie_charts.pdf", format="pdf")
    plt.savefig(f"outputs/png/pie_charts.png", format="png", dpi=300)
    plt.show()


def save_scores(experiment: str, index: str, values: dict) -> None:
    """Log scores for individual models in the corresponding csv file"""
    llms = [
        "BERT",
        "RoBERTa",
        "SetFit-MiniLM",
        "SetFit-mpnet",
        "FLAN-T5-small",
        "FLAN-T5-base",
    ]
    models = ["NB", "LR", "KNN", "SVM", "XGBoost", "LightGBM"]

    Path(f"outputs/csv/").mkdir(parents=True, exist_ok=True)

    file = Path(f"outputs/csv/{experiment}.csv")
    if file.is_file():
        scores = pd.read_csv(f"outputs/csv/{experiment}.csv", index_col=0)
        scores.loc[index] = values
    else:
        if index in llms:
            scores = pd.DataFrame(
                index=llms,
                columns=list(SCORING.keys()) + ["training_time", "inference_time"],
            )
        else:
            scores = pd.DataFrame(
                index=models,
                columns=list(SCORING.keys()) + ["training_time", "inference_time"],
            )
        scores.loc[index] = values

    scores.to_csv(f"outputs/csv/{experiment}.csv")
