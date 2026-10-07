import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader


def make_dataset(
    n_samples=7000,
    n_features=20,
    seed=42
):
    """
    Generate the synthetic binary-classification dataset
    used throughout Gradient Clinic.

    The underlying probability depends on both linear and
    nonlinear combinations of the input features.
    """

    if n_features < 7:
        raise ValueError(
            "n_features must be at least 7."
        )

    rng = np.random.RandomState(seed)

    X = rng.normal(
        size=(n_samples, n_features)
    ).astype(np.float32)

    z = (
        0.85 * X[:, 0]
        - 0.65 * X[:, 1]
        + 0.45 * X[:, 4]
        + 1.10 * X[:, 2] * X[:, 3]
        + 0.95 * np.sin(1.6 * X[:, 5])
        - 0.55 * (X[:, 6] ** 2 - 1.0)
        - 0.65
    )

    probabilities = (
        1.0
        / (1.0 + np.exp(-z))
    )

    y = (
        rng.rand(n_samples)
        < probabilities
    ).astype(np.float32)

    return X, y

def split_dataset(
    X,
    y,
    seed=42
):
    """
    Split data into:

        60% training
        20% validation
        20% test
    """

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.40,
            random_state=seed
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=seed
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def scale_features(
    X_train,
    X_val,
    X_test
):
    """
    Fit the StandardScaler ONLY on training data.

    Validation and test sets are transformed using
    statistics learned from training.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        scaler
    )

def make_train_loader(
    X_train,
    y_train,
    batch_size=128,
    shuffle=True
):
    """
    Build the PyTorch mini-batch training loader.
    """

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


def prepare_data(
    n_samples=7000,
    n_features=20,
    batch_size=128,
    seed=42
):
    """
    Complete data pipeline for Gradient Clinic.

    Returns a dictionary containing the scaled splits,
    training DataLoader and fitted scaler.
    """

    X, y = make_dataset(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(
        X,
        y,
        seed=seed
    )

    (
        X_train,
        X_val,
        X_test,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    train_loader = make_train_loader(
        X_train,
        y_train,
        batch_size=batch_size
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "train_loader": train_loader,

        "scaler": scaler
    }