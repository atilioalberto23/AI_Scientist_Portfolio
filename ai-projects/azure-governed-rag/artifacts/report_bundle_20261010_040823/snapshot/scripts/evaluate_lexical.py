from pathlib import Path

import json
import sys
import time

import numpy as np
import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from azure_governed_rag.search import (
    build_search_client,
    governed_search,
)


def load_config():

    with (
        LAB_ROOT / "config.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def load_golden_queries():

    path = (
        LAB_ROOT
        / "data"
        / "synthetic"
        / "golden_queries.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def recall_at_k(
    expected,
    retrieved,
):

    expected = set(expected)

    if not expected:
        return 1.0 if not retrieved else 0.0

    hits = expected.intersection(
        retrieved
    )

    return (
        len(hits)
        / len(expected)
    )


def precision_at_k(
    expected,
    retrieved,
):

    expected = set(expected)

    if not retrieved:
        return 1.0 if not expected else 0.0

    hits = expected.intersection(
        retrieved
    )

    return (
        len(hits)
        / len(retrieved)
    )


def main():

    config = load_config()
    golden_queries = load_golden_queries()

    search_config = config["search"]

    client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"],
    )

    recalls = []
    precisions = []
    latencies_ms = []

    print(
        "\n"
        "====================================\n"
        " LEXICAL RETRIEVAL EVALUATION\n"
        "====================================\n"
    )

    for case in golden_queries:

        start = time.perf_counter()

        results = governed_search(
            client=client,
            query=case["query"],
            user_groups=case["user_groups"],
            top_k=5,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        retrieved = [
            result["doc_id"]
            for result in results
        ]

        recall = recall_at_k(
            case["expected_doc_ids"],
            retrieved,
        )

        precision = precision_at_k(
            case["expected_doc_ids"],
            retrieved,
        )

        recalls.append(recall)
        precisions.append(precision)
        latencies_ms.append(
            elapsed_ms
        )

        print(
            f"{case['id']} | "
            f"R@5={recall:.2f} | "
            f"P@5={precision:.2f} | "
            f"{elapsed_ms:.1f} ms"
        )

        print(
            "Expected:",
            case["expected_doc_ids"]
        )

        print(
            "Retrieved:",
            retrieved
        )

        print()

    print(
        "===================================="
    )

    print(
        f"Mean Recall@5: "
        f"{np.mean(recalls):.3f}"
    )

    print(
        f"Mean Precision@5: "
        f"{np.mean(precisions):.3f}"
    )

    print(
        f"Latency p95: "
        f"{np.percentile(latencies_ms, 95):.1f} ms"
    )


if __name__ == "__main__":
    main()