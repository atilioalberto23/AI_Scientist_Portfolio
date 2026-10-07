from pathlib import Path
import argparse

import torch
import torch.nn as nn

from gradient_clinic.config import load_config
from gradient_clinic.data import prepare_data
from gradient_clinic.models import MLPClassifier
from gradient_clinic.training import train_model
from gradient_clinic.evaluation import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run a Gradient Clinic experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration."
    )

    return parser.parse_args()


def build_optimizer(
    model,
    training_config
):

    optimizer_name = (
        training_config["optimizer"]
        .lower()
    )

    learning_rate = (
        training_config["learning_rate"]
    )

    weight_decay = (
        training_config.get(
            "weight_decay",
            0.0
        )
    )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if optimizer_name == "sgd":

        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


def main():

    args = parse_args()

    config = load_config(
        args.config
    )

    experiment_name = (
        config["experiment"]["name"]
    )

    seed = config["seed"]

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    print(
        "\n"
        "=====================================\n"
        "       GRADIENT CLINIC\n"
        f"       Experiment: {experiment_name}\n"
        "=====================================\n"
    )

    # =========================================
    # REPRODUCIBILITY
    # =========================================

    torch.manual_seed(seed)

    # =========================================
    # DATA
    # =========================================

    print("Preparing data...")

    data = prepare_data(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        batch_size=data_config["batch_size"],
        seed=seed,
        train_limit=data_config.get(
            "train_limit"
        )
    )

    print(
        "Train:",
        data["X_train"].shape
    )

    print(
        "Validation:",
        data["X_val"].shape
    )

    print(
        "Test:",
        data["X_test"].shape
    )

    # =========================================
    # MODEL
    # =========================================

    model = MLPClassifier(
        n_features=data_config["n_features"],

        hidden_layers=
        model_config["hidden_layers"],

        dropout=
        model_config.get(
            "dropout",
            0.0
        ),

        batch_norm=
        model_config.get(
            "batch_norm",
            True
        )
    )

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTrainable parameters:",
        n_parameters
    )

    # =========================================
    # LOSS
    # =========================================

    loss_fn = nn.BCEWithLogitsLoss()

    # =========================================
    # OPTIMIZER
    # =========================================

    optimizer = build_optimizer(
        model,
        training_config
    )

    # =========================================
    # TRAIN
    # =========================================

    print("\nTraining...")

    history = train_model(
        model=model,

        train_loader=
            data["train_loader"],

        X_val=
            data["X_val"],

        y_val=
            data["y_val"],

        optimizer=optimizer,
        loss_fn=loss_fn,

        epochs=
            training_config["epochs"]
    )

    # =========================================
    # TEST
    # =========================================

    result = evaluate_model(
        model=model,
        X=data["X_test"],
        y=data["y_test"]
    )

    metrics = result["metrics"]

    print(
        "\n"
        "============================\n"
        "TEST METRICS\n"
        "============================"
    )

    for name, value in metrics.items():

        print(
            f"{name:12s}: {value:.4f}"
        )

    # =========================================
    # SAVE HISTORY
    # =========================================

    history_path = (
        METRICS_DIR
        / f"{experiment_name}_history.csv"
    )

    history.to_csv(
        history_path,
        index=False
    )

    # =========================================
    # SAVE TEST METRICS
    # =========================================

    test_metrics_path = (
        METRICS_DIR
        / f"{experiment_name}_test.csv"
    )

    import pandas as pd

    pd.DataFrame(
        [metrics]
    ).to_csv(
        test_metrics_path,
        index=False
    )

    # =========================================
    # SAVE MODEL
    # =========================================

    model_path = (
        MODELS_DIR
        / f"{experiment_name}_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        "\nSaved:"
    )

    print(history_path)
    print(test_metrics_path)
    print(model_path)

    print(
        "\nExperiment completed.\n"
    )


if __name__ == "__main__":
    main()