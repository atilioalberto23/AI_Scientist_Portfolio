from pathlib import Path

import torch
import torch.nn as nn

import argparse

from gradient_clinic.data import prepare_data
from gradient_clinic.models import TinyNet
from gradient_clinic.training import train_model
from gradient_clinic.evaluation import evaluate_model
from gradient_clinic.config import load_config


# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run a Gradient Clinic experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration."
    )

    return parser.parse_args()


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

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
        "       Healthy Training Run\n"
        "=====================================\n"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(seed)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("Preparing data...")

    data = prepare_data(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        batch_size=data_config["batch_size"],
        seed=seed
    )

    print(
        "Train shape:",
        data["X_train"].shape
    )

    print(
        "Validation shape:",
        data["X_val"].shape
    )

    print(
        "Test shape:",
        data["X_test"].shape
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = TinyNet(
        n_features=data_config["n_features"],
        hidden_1=model_config["hidden_1"],
        hidden_2=model_config["hidden_2"],
        dropout_1=model_config["dropout_1"],
        dropout_2=model_config["dropout_2"]
    )

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        n_parameters
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer_name = (
        training_config["optimizer"].lower()
    )

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"]
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining...\n")

    history = train_model(
        model=model,
        train_loader=data["train_loader"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=training_config["epochs"]
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\nEvaluating test set...")

    test_result = evaluate_model(
        model=model,
        X=data["X_test"],
        y=data["y_test"]
    )

    metrics = test_result["metrics"]

    print(
        "\n"
        "============================\n"
        "TEST METRICS\n"
        "============================"
    )

    for metric_name, value in metrics.items():

        print(
            f"{metric_name:10s}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
            METRICS_DIR
            / f"{experiment_name}_history.csv"
    )

    history.to_csv(
        history_path,
        index=False
    )

    print(
        "\nTraining history saved to:"
    )

    print(history_path)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
            MODELS_DIR
            / f"{experiment_name}_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        "\nModel saved to:"
    )

    print(model_path)

    print(
        "\nGradient Clinic run completed."
    )


if __name__ == "__main__":
    main()