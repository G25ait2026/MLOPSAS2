"""
train.py
Model loading, Trainer setup, and training loop.
Logs all metrics to Weights & Biases.

Usage:
    python train.py
"""

import os

import wandb
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import utils
from data import encode_data, load_all_genres, split_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 128
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
SAVED_MODEL_DIR = "distilbert-goodreads-genres"

WANDB_PROJECT = "distilbert-goodreads-genres"
WANDB_RUN_NAME = "distilbert-run-1"

HYPERPARAMS = {
    "model": MODEL_NAME,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "max_length": MAX_LENGTH,
    "dataset": "UCSD Goodreads",
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=HYPERPARAMS["epochs"],
        per_device_train_batch_size=HYPERPARAMS["batch_size"],
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        learning_rate=HYPERPARAMS["learning_rate"],
    )


def train(
    model,
    train_dataset,
    test_dataset,
    training_args: TrainingArguments,
) -> Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=utils.compute_metrics,
    )
    trainer.train()
    return trainer


def main():
    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and prepare data
    genre_reviews_dict = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    tokenizer, train_dataset, test_dataset = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(utils.id2label),
    ).to(device)

    # Initialise W&B
    wandb.init(
        project=WANDB_PROJECT,
        entity="g25ait2026-charantej",
        name=WANDB_RUN_NAME,
        config=HYPERPARAMS,
    )

    # Train
    training_args = build_training_args()
    trainer = train(model, train_dataset, test_dataset, training_args)

    # Save model and tokenizer locally
    trainer.save_model(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)
    print(f"Model and tokenizer saved to: {SAVED_MODEL_DIR}")

    wandb.finish()
    return trainer, tokenizer, test_dataset


if __name__ == "__main__":
    main()
