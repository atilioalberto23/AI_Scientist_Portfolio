from pathlib import Path

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = LAB_ROOT / "artifacts"


SCRIPTS_TO_RUN = [
    "smoke_llm.py",
    "smoke_embeddings.py",
    "search_text.py",
    "search_vector.py",
    "search_hybrid.py",
    "evaluate_lexical.py",
    "evaluate_retrieval.py",
    "run_report_cases.py",
]


PACKAGE_NAMES = [
    "openai",
    "azure-identity",
    "azure-search-documents",
    "PyYAML",
    "numpy",
]


SECRET_WORDS = (
    "key",
    "secret",
    "token",
    "password",
    "credential",
)


def sanitize(value):

    if isinstance(value, dict):

        result = {}

        for key, item in value.items():

            if any(
                word in key.lower()
                for word in SECRET_WORDS
            ):
                result[key] = "***REDACTED***"

            else:
                result[key] = sanitize(item)

        return result

    if isinstance(value, list):

        return [
            sanitize(item)
            for item in value
        ]

    return value


def run_command(
    command,
    log_path,
):

    start = time.perf_counter()

    process = subprocess.run(
        command,
        cwd=LAB_ROOT,
        capture_output=True,
        text=True,
    )

    elapsed = (
        time.perf_counter() - start
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "$ "
            + " ".join(command)
            + "\n\n"
        )

        file.write(
            "===== STDOUT =====\n\n"
        )

        file.write(
            process.stdout
        )

        file.write(
            "\n\n===== STDERR =====\n\n"
        )

        file.write(
            process.stderr
        )

        file.write(
            "\n\n===== EXECUTION =====\n"
        )

        file.write(
            f"return_code: "
            f"{process.returncode}\n"
        )

        file.write(
            f"duration_seconds: "
            f"{elapsed:.3f}\n"
        )

    return {
        "command": command,
        "return_code":
            process.returncode,
        "duration_seconds":
            round(elapsed, 3),
        "log":
            log_path.name,
    }


def write_environment(path):

    packages = {}

    for package in PACKAGE_NAMES:

        try:
            packages[package] = (
                importlib.metadata.version(
                    package
                )
            )

        except (
            importlib.metadata
            .PackageNotFoundError
        ):
            packages[package] = None

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            environment,
            file,
            indent=2,
        )


def capture_git_metadata(path):

    metadata = {}

    commands = {
        "commit": [
            "git",
            "rev-parse",
            "HEAD",
        ],
        "branch": [
            "git",
            "branch",
            "--show-current",
        ],
        "status": [
            "git",
            "status",
            "--short",
        ],
    }

    for name, command in commands.items():

        try:

            result = subprocess.run(
                command,
                cwd=LAB_ROOT,
                capture_output=True,
                text=True,
            )

            metadata[name] = (
                result.stdout.strip()
            )

        except FileNotFoundError:

            metadata[name] = (
                "git unavailable"
            )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


def copy_tree_if_exists(
    source,
    destination,
):

    if source.exists():

        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".DS_Store",
            ),
        )


def copy_file_if_exists(
    source,
    destination,
):

    if source.exists():

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--with-revocation",
        action="store_true",
        help=(
            "Run the revocation test. "
            "This temporarily modifies "
            "an indexed document."
        ),
    )

    args = parser.parse_args()

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    ARTIFACTS_DIR.mkdir(
        exist_ok=True
    )

    staging = (
        ARTIFACTS_DIR
        / f"report_bundle_{timestamp}"
    )

    logs_dir = (
        staging / "logs"
    )

    snapshot_dir = (
        staging / "snapshot"
    )

    logs_dir.mkdir(
        parents=True
    )

    snapshot_dir.mkdir(
        parents=True
    )

    executions = []

    print(
        "\n"
        "====================================\n"
        " BUILDING REPORT BUNDLE\n"
        "====================================\n"
    )

    for script_name in SCRIPTS_TO_RUN:

        script = (
            LAB_ROOT
            / "scripts"
            / script_name
        )

        if not script.exists():

            executions.append(
                {
                    "command": [
                        script_name
                    ],
                    "status": "missing",
                }
            )

            print(
                f"SKIP  {script_name}"
            )

            continue

        print(
            f"RUN   {script_name}"
        )

        log_path = (
            logs_dir
            / f"{script.stem}.txt"
        )

        result = run_command(
            [
                sys.executable,
                str(script),
            ],
            log_path,
        )

        executions.append(
            result
        )

        if result["return_code"] == 0:

            print(
                f"PASS  {script_name}"
            )

        else:

            print(
                f"FAIL  {script_name}"
            )

    # --------------------------------
    # Optional revocation experiment
    # --------------------------------

    if args.with_revocation:

        revocation_script = (
            LAB_ROOT
            / "scripts"
            / "test_revocation.py"
        )

        restore_script = (
            LAB_ROOT
            / "scripts"
            / "upload_vector_documents.py"
        )

        if revocation_script.exists():

            print(
                "RUN   test_revocation.py"
            )

            result = run_command(
                [
                    sys.executable,
                    str(revocation_script),
                ],
                logs_dir
                / "test_revocation.txt",
            )

            executions.append(
                result
            )

            print(
                "RESTORE vector documents"
            )

            if restore_script.exists():

                restore = run_command(
                    [
                        sys.executable,
                        str(restore_script),
                    ],
                    logs_dir
                    / "restore_after_revocation.txt",
                )

                executions.append(
                    restore
                )

    # --------------------------------
    # Snapshot source code
    # --------------------------------

    copy_tree_if_exists(
        LAB_ROOT
        / "src"
        / "azure_governed_rag",
        snapshot_dir
        / "src"
        / "azure_governed_rag",
    )

    copy_tree_if_exists(
        LAB_ROOT
        / "scripts",
        snapshot_dir
        / "scripts",
    )

    copy_tree_if_exists(
        LAB_ROOT
        / "data"
        / "synthetic",
        snapshot_dir
        / "data"
        / "synthetic",
    )

    # --------------------------------
    # Existing artifacts
    # --------------------------------

    artifact_snapshot = (
        snapshot_dir
        / "artifacts"
    )

    artifact_snapshot.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in [
        "index_schema.json",
        "rag_report_cases.json",
        "search_report.md",
        "architecture.md",
        "azure_map.md",
        "trace_sample.json",
    ]:

        copy_file_if_exists(
            ARTIFACTS_DIR
            / filename,
            artifact_snapshot
            / filename,
        )

    # --------------------------------
    # Sanitized configuration
    # --------------------------------

    config_path = (
        LAB_ROOT / "config.yaml"
    )

    if config_path.exists():

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            config = yaml.safe_load(
                file
            )

        clean_config = sanitize(
            config
        )

        with (
            snapshot_dir
            / "config_sanitized.yaml"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                clean_config,
                file,
                sort_keys=False,
            )

    copy_file_if_exists(
        LAB_ROOT
        / "config.example.yaml",
        snapshot_dir
        / "config.example.yaml",
    )

    copy_file_if_exists(
        LAB_ROOT / "README.md",
        snapshot_dir / "README.md",
    )

    # --------------------------------
    # Environment + git
    # --------------------------------

    write_environment(
        staging
        / "environment.json"
    )

    capture_git_metadata(
        staging
        / "git_metadata.json"
    )

    # --------------------------------
    # Manifest
    # --------------------------------

    manifest = {
        "project":
            "Azure Governed RAG",
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "executions":
            executions,
        "revocation_test_requested":
            args.with_revocation,
    }

    with (
        staging
        / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    # --------------------------------
    # ZIP
    # --------------------------------

    zip_base = (
        ARTIFACTS_DIR
        / (
            "azure_governed_rag_results_"
            + timestamp
        )
    )

    zip_path = shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=staging,
    )

    print(
        "\n"
        "===================================="
    )

    print(
        "REPORT BUNDLE READY"
    )

    print(
        "====================================\n"
    )

    print(
        zip_path
    )


if __name__ == "__main__":
    main()