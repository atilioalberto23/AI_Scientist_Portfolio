#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    "healthy.yaml",
    "overfit.yaml",
    "overfit_dropout.yaml",
    "overfit_weight_decay.yaml",
    "lr_slow.yaml",
    "lr_normal.yaml",
    "lr_aggressive.yaml",
]


def main():

    print(
        "\n"
        "========================================\n"
        "       GRADIENT CLINIC v0.1\n"
        "       Experimental Suite\n"
        "========================================\n"
    )

    start_total = time.time()

    completed = []
    failed = []

    for index, config_name in enumerate(
        EXPERIMENTS,
        start=1
    ):

        config_path = (
            PROJECT_ROOT
            / "configs"
            / config_name
        )

        print(
            f"\n[{index}/{len(EXPERIMENTS)}] "
            f"Running {config_name}"
        )

        print("-" * 50)

        start = time.time()

        result = subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "train.py"
                ),
                "--config",
                str(config_path)
            ],
            cwd=PROJECT_ROOT
        )

        elapsed = time.time() - start

        if result.returncode == 0:

            completed.append(
                config_name
            )

            print(
                f"\n✓ {config_name} completed "
                f"in {elapsed:.1f} s"
            )

        else:

            failed.append(
                config_name
            )

            print(
                f"\n✗ {config_name} failed"
            )

    total_time = time.time() - start_total

    print(
        "\n"
        "========================================\n"
        "       EXPERIMENT SUMMARY\n"
        "========================================"
    )

    print(
        f"\nCompleted: {len(completed)}"
    )

    for name in completed:
        print(f"  ✓ {name}")

    if failed:

        print(
            f"\nFailed: {len(failed)}"
        )

        for name in failed:
            print(f"  ✗ {name}")

    print(
        f"\nTotal runtime: "
        f"{total_time:.1f} s"
    )

    if failed:
        sys.exit(1)

    print(
        "\nAll Gradient Clinic experiments "
        "completed successfully.\n"
    )


if __name__ == "__main__":
    main()