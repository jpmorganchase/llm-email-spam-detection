"""Functions related to the training and evaluation pipeline"""

from transformers import (
    AutoModelForSequenceClassification,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer,
    AutoTokenizer,
    Trainer,
)
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import evaluate
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate

import copy
import time
import pandas as pd
import pickle
import torch

from src.spamdetection.preprocessing import get_dataset, train_val_test_split
from src.spamdetection.utils import (
    SCORING,
    set_seed,
    plot_loss,
    plot_scores,
    save_scores,
)
from src.spamdetection.transforms import transform_df, encode_df, tokenize, init_nltk


MODELS = {
    "NB": (MultinomialNB(), 1000),
    "LR": (LogisticRegression(), 500),
    "KNN": (KNeighborsClassifier(n_neighbors=1), 150),
    "SVM": (SVC(kernel="sigmoid", gamma=1.0), 3000),
    "XGBoost": (XGBClassifier(learning_rate=0.01, n_estimators=150), 2000),
    "LightGBM": (LGBMClassifier(learning_rate=0.1, num_leaves=20), 3000),
}


LLMS = {
    "RoBERTa": (
        AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        ),
        AutoTokenizer.from_pretrained("roberta-base"),
    ),
    "SetFit-mpnet": (
        SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        None,
    ),
    "FLAN-T5-base": (
        AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base"),
        AutoTokenizer.from_pretrained("google/flan-t5-base"),
    ),
}


class EvalOnTrainCallback(TrainerCallback):
    """Custom callback to evaluate on the training set during training."""

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_train = copy.deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_train


def get_trainer(model, dataset, tokenizer=None):
    """Return a trainer object for transformer models."""

    def compute_metrics(y_pred):
        """Computer metrics during training."""
        logits, labels = y_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluate.load("f1").compute(
            predictions=predictions, references=labels, average="macro"
        )

    if type(model).__name__ == "SetFitModel":
        trainer = SetFitTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            loss_class=CosineSimilarityLoss,
            metric="f1",
            batch_size=16,
            num_iterations=20,
            num_epochs=3,
        )
        return trainer

    elif "T5" in type(model).__name__ or "FLAN" in type(model).__name__:

        def compute_metrics_t5(y_pred, verbose=0):
            """Computer metrics during training for T5-like models."""
            predictions, labels = y_pred

            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            # Replace -100 with pad_token_id to decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions = [
                1 if "spam" in predictions[i] else 0 for i in range(len(predictions))
            ]
            labels = [1 if "spam" in labels[i] else 0 for i in range(len(labels))]

            result = evaluate.load("f1").compute(
                predictions=predictions, references=labels, average="macro"
            )
            return result

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir="experiments",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=5,
            predict_with_generate=True,
            fp16=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=5,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_t5,
        )
        trainer.add_callback(EvalOnTrainCallback(trainer))
        return trainer

    else:
        training_args = TrainingArguments(
            output_dir="experiments",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(EvalOnTrainCallback(trainer))
        return trainer


def predict(trainer, model, dataset, tokenizer=None):
    """Convert the predict function to specific classes to unify the API."""
    if type(model).__name__ == "SetFitModel":
        return model(dataset["text"])

    elif "T5" in type(model).__name__:
        predictions = trainer.predict(dataset)
        predictions = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        predictions = [
            1 if "spam" in predictions[i] else 0 for i in range(len(predictions))
        ]

        return predictions

    else:
        return trainer.predict(dataset).predictions.argmax(axis=-1)


def train_llms(seeds, datasets, train_sizes, test_set="test"):
    """Train all the large language models."""
    for seed in list(seeds):
        set_seed(seed)

        for dataset_name in list(datasets):

            for train_size in train_sizes:
                # Get metrics
                scores = pd.DataFrame(
                    index=list(LLMS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                # Main loop
                df = get_dataset(dataset_name)
                _, dataset = train_val_test_split(
                    df, train_size=train_size, has_val=True
                )

                # Name experiment
                experiment = (
                    f"llm_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"
                )

                # Train, evaluate, test
                for model_name, (model, tokenizer) in LLMS.items():
                    tokenized_dataset = tokenize(dataset, tokenizer)
                    trainer = get_trainer(model, tokenized_dataset, tokenizer)

                    # Train model
                    start = time.time()
                    train_result = trainer.train()
                    end = time.time()
                    scores.loc[model_name]["training_time"] = end - start
                    if "SetFit" not in model_name:
                        log = pd.DataFrame(trainer.state.log_history)
                        log.to_csv(f"outputs/csv/loss_{model_name}_{experiment}.csv")
                        plot_loss(experiment, dataset_name, model_name)

                    # Test model
                    start = time.time()
                    predictions = predict(
                        trainer, model, tokenized_dataset[test_set], tokenizer
                    )
                    end = time.time()

                    for score_name, score_fn in SCORING.items():
                        scores.loc[model_name][score_name] = score_fn(
                            dataset[test_set]["label"], predictions
                        )

                    scores.loc[model_name]["inference_time"] = end - start
                    save_scores(
                        experiment, model_name, scores.loc[model_name].to_dict()
                    )

                # Display scores
                plot_scores(experiment, dataset_name)
                print(scores)


def train_baselines(seeds, datasets, train_sizes, test_set="test"):
    """Train all the baseline models."""
    init_nltk()

    for seed in list(seeds):
        set_seed(seed)

        for dataset_name in list(datasets):

            for train_size in train_sizes:
                # Create list of metrics
                scores = pd.DataFrame(
                    index=list(MODELS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                # Main loop
                df = get_dataset(dataset_name)
                df = transform_df(df)
                (df_train, df_val, df_test), _ = train_val_test_split(
                    df, train_size=train_size, has_val=True
                )

                # Name experiment
                experiment = (
                    f"ml_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"
                )

                # Cross-validate and test every model
                for model_name, (model, max_iter) in MODELS.items():
                    # Encode the dataset
                    encoder = TfidfVectorizer(max_features=max_iter)
                    X_train, y_train, encoder = encode_df(df_train, encoder)
                    X_test, y_test, encoder = encode_df(df_test, encoder)

                    # Evaluate model with cross-validation
                    if test_set == "val":
                        cv = cross_validate(
                            model,
                            X_train,
                            y_train,
                            scoring=list(SCORING.keys()),
                            cv=5,
                            n_jobs=-1,
                        )
                        for score_name, score_fn in SCORING.items():
                            scores.loc[model_name][score_name] = cv[
                                f"test_{score_name}"
                            ].mean()

                    # Evaluate model on test set
                    if test_set == "test":
                        start = time.time()
                        model.fit(X_train, y_train)
                        end = time.time()
                        scores.loc[model_name]["training_time"] = end - start

                        start = time.time()
                        y_pred = model.predict(X_test)
                        end = time.time()

                        scores.loc[model_name]["inference_time"] = end - start
                        for score_name, score_fn in SCORING.items():
                            scores.loc[model_name][score_name] = score_fn(
                                y_pred, y_test
                            )

                    save_scores(
                        experiment, model_name, scores.loc[model_name].to_dict()
                    )

                # Display scores
                plot_scores(experiment, dataset_name)
                print(scores)
