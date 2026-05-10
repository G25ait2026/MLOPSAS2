"""
eval.py
Evaluation on the test set, metrics logging to W&B, and artifact upload.

Usage (after training):
    python eval.py

This script can be run standalone if a saved model already exists, or it can
be imported and called from train.py via run_evaluation().
"""

import json
import os

import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

import utils
from data import encode_data, load_all_genres, split_data
from train import (
    SAVED_MODEL_DIR,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    build_training_args,
    MAX_LENGTH,
)

EVAL_REPORT_PATH = "eval_report.json"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_evaluation(trainer: Trainer, test_dataset, test_labels: list[str]) -> dict:
    """
    Run trainer.evaluate(), log metrics to W&B, save a classification report,
    and upload it as a W&B Artifact.

    Returns the eval_results dict.
    """
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Log final metrics explicitly
    wandb.log(
        {
            "final/loss": eval_results.get("eval_loss"),
            "final/accuracy": eval_results.get("eval_accuracy"),
            "final/f1": eval_results.get("eval_f1"),
        }
    )

    # Detailed classification report
    predicted = trainer.predict(test_dataset).predictions.argmax(-1)
    predicted_labels = [utils.id2label[i] for i in predicted.flatten().tolist()]

    report = classification_report(
        test_labels,
        predicted_labels,
        target_names=list(utils.id2label.values()),
        output_dict=True,
    )

    with open(EVAL_REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Classification report saved to: {EVAL_REPORT_PATH}")

    # Upload report as a versioned W&B Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(EVAL_REPORT_PATH)
    wandb.log_artifact(artifact)
    print("Evaluation artifact uploaded to W&B.")

    return eval_results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data (needed for label maps and test set)
    genre_reviews_dict = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    tokenizer, train_dataset, test_dataset = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    # Load the saved model
    model = DistilBertForSequenceClassification.from_pretrained(SAVED_MODEL_DIR).to(device)

    # Initialise a W&B run for evaluation
    wandb.init(
        project=WANDB_PROJECT,
        entity="g25ait2026-charantej",
        name=f"{WANDB_RUN_NAME}-eval",
    )

    # Build a minimal Trainer just for evaluation (no training)
    training_args = build_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=utils.compute_metrics,
    )

    run_evaluation(trainer, test_dataset, test_labels)
    wandb.finish()


if __name__ == "__main__":
    main()
