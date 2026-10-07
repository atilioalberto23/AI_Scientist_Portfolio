import numpy as np
import pandas as pd
import torch

from gradient_clinic.diagnostics import (
    gradient_norm,
    has_nonfinite_gradients,
    has_nonfinite_parameters
)

from gradient_clinic.evaluation import (
    classification_metrics
)


def train_model(
    model,
    train_loader,
    X_val,
    y_val,
    optimizer,
    loss_fn,
    epochs=30
):
    """
    Train a binary classification neural network.

    During every epoch the function records:

    - training loss
    - validation loss
    - gradient norm
    - ROC-AUC
    - PR-AUC
    - Brier score
    - learning rate
    - numerical health checks

    The model is modified in place.

    Returns
    -------
    pandas.DataFrame
        Training history.
    """

    history = []

    # -----------------------------------------
    # Validation data -> tensors only once
    # -----------------------------------------

    if torch.is_tensor(X_val):
        X_val_tensor = X_val.float()

    else:
        X_val_tensor = torch.tensor(
            X_val,
            dtype=torch.float32
        )

    if torch.is_tensor(y_val):
        y_val_tensor = y_val.float()

    else:
        y_val_tensor = torch.tensor(
            y_val,
            dtype=torch.float32
        )

    # =========================================
    # TRAINING LOOP
    # =========================================

    for epoch in range(epochs):

        # =====================================
        # TRAINING PHASE
        # =====================================

        model.train()

        batch_losses = []
        batch_gradient_norms = []

        nonfinite_gradient_detected = False
        nonfinite_parameter_detected = False

        for xb, yb in train_loader:

            # ---------------------------------
            # 1. Clear previous gradients
            # ---------------------------------

            optimizer.zero_grad()

            # ---------------------------------
            # 2. Forward pass
            # ---------------------------------

            logits = model(xb)

            # ---------------------------------
            # 3. Compute loss
            # ---------------------------------

            loss = loss_fn(
                logits,
                yb
            )

            # ---------------------------------
            # 4. Backpropagation
            # ---------------------------------

            loss.backward()

            # ---------------------------------
            # 5. Diagnostics BEFORE update
            # ---------------------------------

            grad_norm = gradient_norm(
                model
            )

            if has_nonfinite_gradients(model):
                nonfinite_gradient_detected = True

            # ---------------------------------
            # 6. Update model parameters
            # ---------------------------------

            optimizer.step()

            if has_nonfinite_parameters(model):
                nonfinite_parameter_detected = True

            # ---------------------------------
            # 7. Store batch information
            # ---------------------------------

            batch_losses.append(
                loss.item()
            )

            batch_gradient_norms.append(
                grad_norm
            )

        # =====================================
        # VALIDATION PHASE
        # =====================================

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )

            val_loss = loss_fn(
                val_logits,
                y_val_tensor
            )

            val_probabilities = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        # -------------------------------------
        # Validation metrics
        # -------------------------------------

        metrics = classification_metrics(
            y_val,
            val_probabilities
        )

        # -------------------------------------
        # Current learning rate
        # -------------------------------------

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # =====================================
        # SAVE EPOCH
        # =====================================

        history.append({
            "epoch":
                epoch + 1,

            "train_loss":
                np.mean(batch_losses),

            "val_loss":
                val_loss.item(),

            "gradient_norm":
                np.mean(batch_gradient_norms),

            "val_roc_auc":
                metrics["roc_auc"],

            "val_pr_auc":
                metrics["pr_auc"],

            "val_brier":
                metrics["brier"],

            "learning_rate":
                current_lr,

            "nonfinite_gradients":
                nonfinite_gradient_detected,

            "nonfinite_parameters":
                nonfinite_parameter_detected
        })

    return pd.DataFrame(history)