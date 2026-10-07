from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT REGISTRY
# ============================================================

EXPERIMENTS = {
    "healthy": "Healthy baseline",

    "overfit": "Unregularized",
    "overfit_dropout": "Dropout",
    "overfit_weight_decay": "Weight decay",

    "lr_slow": r"Slow ($\eta=10^{-5}$)",
    "lr_normal": r"Reference ($\eta=10^{-3}$)",
    "lr_aggressive": r"Aggressive ($\eta=10^{-1}$)",
}


# ============================================================
# VISUAL SYSTEM
# ============================================================

# Inferno is sampled rather than using its extreme ends.
# This avoids nearly-black and nearly-yellow curves.
INFERNO_2 = sns.color_palette(
    "inferno",
    n_colors=4
)[1:3]

INFERNO_3 = sns.color_palette(
    "inferno",
    n_colors=5
)[1:4]


COLORS = {
    "train": INFERNO_2[0],
    "validation": INFERNO_2[1],

    "overfit": INFERNO_3[0],
    "dropout": INFERNO_3[1],
    "weight_decay": INFERNO_3[2],

    "lr_slow": INFERNO_3[0],
    "lr_normal": INFERNO_3[1],
    "lr_aggressive": INFERNO_3[2],
}


def configure_style():
    """
    Configure the global visual language used by all report figures.

    The design aims for publication-style plots:
    strong typography, restrained grids, high contrast,
    consistent spacing and the Inferno palette.
    """

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans"
    )

    plt.rcParams.update({

        # --------------------------------------
        # Figure
        # --------------------------------------
        "figure.figsize": (7.4, 4.6),
        "figure.dpi": 120,
        "savefig.dpi": 400,

        # --------------------------------------
        # Typography
        # --------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Arial",
            "Helvetica"
        ],

        "font.size": 10.5,

        "axes.titlesize": 14,
        "axes.titleweight": "bold",

        "axes.labelsize": 11,
        "axes.labelweight": "semibold",

        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,

        "legend.fontsize": 9.5,
        "legend.title_fontsize": 9.5,

        # --------------------------------------
        # Lines
        # --------------------------------------
        "lines.linewidth": 2.25,

        # --------------------------------------
        # Axes
        # --------------------------------------
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#2A2A2A",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#141414",

        # --------------------------------------
        # Grid
        # --------------------------------------
        "grid.alpha": 0.16,
        "grid.linewidth": 0.7,

        # --------------------------------------
        # Legend
        # --------------------------------------
        "legend.frameon": False,

        # --------------------------------------
        # Export
        # --------------------------------------
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,

        # Better text embedding in PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# DATA ACCESS
# ============================================================

def load_history(name):

    path = (
        METRICS_DIR
        / f"{name}_history.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing history file: {path}"
        )

    return pd.read_csv(path)


def load_test(name):

    path = (
        METRICS_DIR
        / f"{name}_test.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing test file: {path}"
        )

    return pd.read_csv(path).iloc[0]


# ============================================================
# FIGURE UTILITIES
# ============================================================

def make_figure(
    figsize=(7.4, 4.6)
):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    return fig, ax


def editorial_axes(
    ax,
    xlabel,
    ylabel,
    title=None
):
    """
    Apply the common publication treatment to an axis.
    """

    ax.set_xlabel(
        xlabel,
        labelpad=8
    )

    ax.set_ylabel(
        ylabel,
        labelpad=8
    )

    if title is not None:

        ax.set_title(
            title,
            loc="left",
            pad=14
        )

    sns.despine(
        ax=ax,
        top=True,
        right=True
    )

    ax.grid(
        axis="both",
        alpha=0.16
    )

    ax.set_axisbelow(True)

    ax.margins(x=0.01)


def save_figure(
    fig,
    stem
):
    """
    Save every publication figure in both:
      - vector PDF for the report
      - high-resolution PNG for README/web
    """

    pdf_path = (
        FIGURES_DIR
        / f"{stem}.pdf"
    )

    png_path = (
        FIGURES_DIR
        / f"{stem}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    fig.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {pdf_path}"
    )

    print(
        f"Saved: {png_path}"
    )


def add_best_epoch(
    ax,
    history
):

    best_index = (
        history["val_loss"]
        .idxmin()
    )

    best_row = history.loc[
        best_index
    ]

    best_epoch = int(
        best_row["epoch"]
    )

    best_loss = float(
        best_row["val_loss"]
    )

    ax.axvline(
        best_epoch,
        color="#363636",
        linestyle="--",
        linewidth=1.1,
        alpha=0.65
    )

    ax.scatter(
        [best_epoch],
        [best_loss],
        s=42,
        color="#161616",
        zorder=10
    )

    ax.annotate(
        f"Best validation\nEpoch {best_epoch}",
        xy=(
            best_epoch,
            best_loss
        ),
        xytext=(12, 16),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="semibold",
        color="#262626",
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.8,
            "color": "#555555"
        }
    )


# ============================================================
# SUMMARY ENGINE
# ============================================================

def experiment_summary(name):

    history = load_history(name)
    test = load_test(name)

    best_index = (
        history["val_loss"]
        .idxmin()
    )

    best_row = history.loc[
        best_index
    ]

    final_row = history.iloc[-1]

    gap = (
        final_row["val_loss"]
        - final_row["train_loss"]
    )

    return {

        "experiment":
            name,

        "best_epoch":
            int(best_row["epoch"]),

        "best_val_loss":
            float(best_row["val_loss"]),

        "final_train_loss":
            float(final_row["train_loss"]),

        "final_val_loss":
            float(final_row["val_loss"]),

        "final_generalization_gap":
            float(gap),

        "max_gradient_norm":
            float(
                history[
                    "gradient_norm"
                ].max()
            ),

        "mean_gradient_norm":
            float(
                history[
                    "gradient_norm"
                ].mean()
            ),

        "test_roc_auc":
            float(test["roc_auc"]),

        "test_pr_auc":
            float(test["pr_auc"]),

        "test_brier":
            float(test["brier"]),

        "nonfinite_gradients":
            bool(
                history[
                    "nonfinite_gradients"
                ].any()
            ),

        "nonfinite_parameters":
            bool(
                history[
                    "nonfinite_parameters"
                ].any()
            ),
    }


# ============================================================
# FIGURE 1
# HEALTHY BASELINE — LOSS
# ============================================================

def plot_healthy_loss():

    history = load_history(
        "healthy"
    )

    fig, ax = make_figure()

    sns.lineplot(
        data=history,
        x="epoch",
        y="train_loss",
        ax=ax,
        color=COLORS["train"],
        linewidth=2.4,
        label="Training"
    )

    sns.lineplot(
        data=history,
        x="epoch",
        y="val_loss",
        ax=ax,
        color=COLORS["validation"],
        linewidth=2.4,
        label="Validation"
    )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel="Binary cross-entropy",
        title="Healthy training dynamics"
    )

    ax.legend(
        loc="best"
    )

    save_figure(
        fig,
        "fig01_healthy_loss"
    )


# ============================================================
# FIGURE 2
# HEALTHY BASELINE — PERFORMANCE
# ============================================================

def plot_healthy_metrics():

    history = load_history(
        "healthy"
    )

    fig, ax = make_figure()

    sns.lineplot(
        data=history,
        x="epoch",
        y="val_roc_auc",
        ax=ax,
        color=INFERNO_2[0],
        linewidth=2.4,
        label="ROC-AUC"
    )

    sns.lineplot(
        data=history,
        x="epoch",
        y="val_pr_auc",
        ax=ax,
        color=INFERNO_2[1],
        linewidth=2.4,
        label="PR-AUC"
    )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel="Validation score",
        title="Discriminative performance"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.legend(
        loc="lower right"
    )

    save_figure(
        fig,
        "fig02_healthy_performance"
    )


# ============================================================
# FIGURE 3
# HEALTHY BASELINE — GRADIENTS
# ============================================================

def plot_healthy_gradients():

    history = load_history(
        "healthy"
    )

    fig, ax = make_figure()

    sns.lineplot(
        data=history,
        x="epoch",
        y="gradient_norm",
        ax=ax,
        color=INFERNO_3[1],
        linewidth=2.4
    )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel=r"Mean $\|\nabla_\theta L\|_2$",
        title="Gradient dynamics under healthy optimization"
    )

    save_figure(
        fig,
        "fig03_healthy_gradients"
    )


# ============================================================
# FIGURE 4
# OVERFITTING
# ============================================================

def plot_overfit_train_validation():

    history = load_history(
        "overfit"
    )

    fig, ax = make_figure()

    sns.lineplot(
        data=history,
        x="epoch",
        y="train_loss",
        ax=ax,
        color=COLORS["train"],
        linewidth=2.3,
        label="Training"
    )

    sns.lineplot(
        data=history,
        x="epoch",
        y="val_loss",
        ax=ax,
        color=COLORS["validation"],
        linewidth=2.3,
        label="Validation"
    )

    add_best_epoch(
        ax,
        history
    )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel="Binary cross-entropy",
        title="Patient I · Overfitting"
    )

    ax.legend(
        loc="best"
    )

    save_figure(
        fig,
        "fig04_overfitting"
    )


# ============================================================
# FIGURE 5
# REGULARIZATION
# ============================================================

def plot_regularization_comparison():

    specifications = [
        (
            "overfit",
            "Unregularized",
            COLORS["overfit"]
        ),
        (
            "overfit_dropout",
            "Dropout",
            COLORS["dropout"]
        ),
        (
            "overfit_weight_decay",
            "Weight decay",
            COLORS["weight_decay"]
        ),
    ]

    fig, ax = make_figure()

    for name, label, color in specifications:

        history = load_history(
            name
        )

        sns.lineplot(
            data=history,
            x="epoch",
            y="val_loss",
            ax=ax,
            color=color,
            linewidth=2.25,
            label=label
        )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel="Validation loss",
        title="Regularization reshapes generalization"
    )

    ax.legend(
        loc="best"
    )

    save_figure(
        fig,
        "fig05_regularization"
    )


# ============================================================
# FIGURE 6
# GENERALIZATION GAP
# ============================================================

def plot_generalization_gap():

    specifications = [
        (
            "overfit",
            "Unregularized",
            COLORS["overfit"]
        ),
        (
            "overfit_dropout",
            "Dropout",
            COLORS["dropout"]
        ),
        (
            "overfit_weight_decay",
            "Weight decay",
            COLORS["weight_decay"]
        ),
    ]

    rows = []

    colors = []

    for name, label, color in specifications:

        history = load_history(
            name
        )

        final = history.iloc[-1]

        gap = (
            final["val_loss"]
            - final["train_loss"]
        )

        rows.append({
            "Treatment": label,
            "Generalization gap": gap
        })

        colors.append(
            color
        )

    dataframe = pd.DataFrame(
        rows
    )

    fig, ax = make_figure(
        figsize=(6.8, 4.6)
    )

    sns.barplot(
        data=dataframe,
        x="Treatment",
        y="Generalization gap",
        palette=colors,
        hue="Treatment",
        legend=False,
        ax=ax
    )

    editorial_axes(
        ax,
        xlabel="",
        ylabel=r"$L_{\mathrm{val}} - L_{\mathrm{train}}$",
        title="Final generalization gap"
    )

    for patch in ax.patches:

        height = patch.get_height()

        ax.annotate(
            f"{height:.3f}",
            (
                patch.get_x()
                + patch.get_width() / 2,
                height
            ),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    save_figure(
        fig,
        "fig06_generalization_gap"
    )


# ============================================================
# FIGURE 7
# LEARNING-RATE DYNAMICS
# ============================================================

def plot_learning_rate_loss():

    specifications = [
        (
            "lr_slow",
            EXPERIMENTS["lr_slow"],
            COLORS["lr_slow"]
        ),
        (
            "lr_normal",
            EXPERIMENTS["lr_normal"],
            COLORS["lr_normal"]
        ),
        (
            "lr_aggressive",
            EXPERIMENTS["lr_aggressive"],
            COLORS["lr_aggressive"]
        ),
    ]

    fig, ax = make_figure()

    for name, label, color in specifications:

        history = load_history(
            name
        )

        sns.lineplot(
            data=history,
            x="epoch",
            y="val_loss",
            ax=ax,
            color=color,
            linewidth=2.3,
            label=label
        )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel="Validation loss",
        title="Patient II · Learning-rate pathology"
    )

    ax.legend(
        loc="best"
    )

    save_figure(
        fig,
        "fig07_learning_rate_loss"
    )


# ============================================================
# FIGURE 8
# LEARNING-RATE GRADIENTS
# ============================================================

def plot_learning_rate_gradients():

    specifications = [
        (
            "lr_slow",
            EXPERIMENTS["lr_slow"],
            COLORS["lr_slow"]
        ),
        (
            "lr_normal",
            EXPERIMENTS["lr_normal"],
            COLORS["lr_normal"]
        ),
        (
            "lr_aggressive",
            EXPERIMENTS["lr_aggressive"],
            COLORS["lr_aggressive"]
        ),
    ]

    fig, ax = make_figure()

    for name, label, color in specifications:

        history = load_history(
            name
        )

        sns.lineplot(
            data=history,
            x="epoch",
            y="gradient_norm",
            ax=ax,
            color=color,
            linewidth=2.2,
            label=label
        )

    editorial_axes(
        ax,
        xlabel="Epoch",
        ylabel=r"Mean $\|\nabla_\theta L\|_2$",
        title="Gradient response to learning-rate choice"
    )

    ax.legend(
        loc="best"
    )

    save_figure(
        fig,
        "fig08_learning_rate_gradients"
    )


# ============================================================
# REPORT TABLES
# ============================================================

def build_test_table():

    rows = []

    for name, label in EXPERIMENTS.items():

        test = load_test(
            name
        )

        rows.append({

            "experiment":
                label,

            "roc_auc":
                float(test["roc_auc"]),

            "pr_auc":
                float(test["pr_auc"]),

            "brier":
                float(test["brier"]),
        })

    table = pd.DataFrame(
        rows
    )

    path = (
        METRICS_DIR
        / "experiment_test_summary.csv"
    )

    table.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


def build_experiment_summary():

    summaries = [
        experiment_summary(name)
        for name in EXPERIMENTS
    ]

    dataframe = pd.DataFrame(
        summaries
    )

    csv_path = (
        METRICS_DIR
        / "experiment_summary.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )

    json_path = (
        METRICS_DIR
        / "experiment_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summaries,
            file,
            indent=4
        )

    print(
        f"Saved: {csv_path}"
    )

    print(
        f"Saved: {json_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    configure_style()

    print(
        "\n"
        "========================================\n"
        "      GRADIENT CLINIC v0.1\n"
        "      Publication Asset Builder\n"
        "========================================\n"
    )

    plot_healthy_loss()
    plot_healthy_metrics()
    plot_healthy_gradients()

    plot_overfit_train_validation()
    plot_regularization_comparison()
    plot_generalization_gap()

    plot_learning_rate_loss()
    plot_learning_rate_gradients()

    build_test_table()
    build_experiment_summary()

    print(
        "\n"
        "Publication assets completed.\n"
    )


if __name__ == "__main__":
    main()