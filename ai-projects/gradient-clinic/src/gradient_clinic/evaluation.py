import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


def predict_probabilities(model, X):
    """
    Run the model in inference mode and return probabilities.

    The model outputs logits, so sigmoid is applied here.
    """

    model.eval()

    if not torch.is_tensor(X):
        X = torch.tensor(
            X,
            dtype=torch.float32
        )

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities.cpu().numpy()


def classification_metrics(
    y_true,
    probabilities
):
    """
    Core probabilistic classification metrics.
    """

    return {
        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "brier":
            brier_score_loss(
                y_true,
                probabilities
            )
    }


def evaluate_model(
    model,
    X,
    y
):
    """
    Evaluate a binary classifier.

    Returns both predictions and metrics.
    """

    probabilities = predict_probabilities(
        model,
        X
    )

    metrics = classification_metrics(
        y,
        probabilities
    )

    return {
        "probabilities": probabilities,
        "metrics": metrics
    }