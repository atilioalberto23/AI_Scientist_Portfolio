from pathlib import Path

import json
import sys
import time

import numpy as np
import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from azure_governed_rag.embeddings import (
    build_embedding_client,
    embed_text,
)

from azure_governed_rag.search import (
    build_search_client,
    governed_lexical_search,
    governed_vector_search,
    governed_hybrid_search,
)


TOP_K = 5


def load_config():

    with (LAB_ROOT / "config.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def load_cases():

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


def unique_doc_ids(results):

    ids = []

    for result in results:

        doc_id = result["doc_id"]

        if doc_id not in ids:
            ids.append(doc_id)

    return ids


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

    return len(hits) / len(expected)


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

    return len(hits) / len(retrieved)


def hit_at_k(
    expected,
    retrieved,
):

    expected = set(expected)

    if not expected:
        return 1.0 if not retrieved else 0.0

    return float(
        bool(
            expected.intersection(
                retrieved
            )
        )
    )


def empty_metrics():

    return {
        "recall": [],
        "precision": [],
        "hit": [],
        "latency": [],
    }


def summarize(
    name,
    metrics,
):

    return {
        "mode": name,

        "recall@5":
            np.mean(
                metrics["recall"]
            ),

        "precision@5":
            np.mean(
                metrics["precision"]
            ),

        "hit@5":
            np.mean(
                metrics["hit"]
            ),

        "p95_ms":
            np.percentile(
                metrics["latency"],
                95,
            ),
    }


def add_result(
    metrics,
    expected,
    results,
    elapsed_ms,
):

    retrieved = unique_doc_ids(
        results
    )[:TOP_K]

    metrics["recall"].append(
        recall_at_k(
            expected,
            retrieved,
        )
    )

    metrics["precision"].append(
        precision_at_k(
            expected,
            retrieved,
        )
    )

    metrics["hit"].append(
        hit_at_k(
            expected,
            retrieved,
        )
    )

    metrics["latency"].append(
        elapsed_ms
    )

    return retrieved


def main():

    config = load_config()
    cases = load_cases()

    azure_config = config["azure"]
    search_config = config["search"]

    search_client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"],
    )

    embedding_client = build_embedding_client(
        base_url=azure_config[
            "openai_base_url"
        ]
    )

    lexical_metrics = empty_metrics()
    vector_metrics = empty_metrics()
    hybrid_metrics = empty_metrics()

    embedding_latencies = []

    print(
        "\n"
        "====================================\n"
        " RETRIEVAL BENCHMARK\n"
        "====================================\n"
    )

    for case in cases:

        query = case["query"]
        groups = case["user_groups"]
        expected = case[
            "expected_doc_ids"
        ]

        # ----------------------------
        # Lexical
        # ----------------------------

        start = time.perf_counter()

        lexical = governed_lexical_search(
            client=search_client,
            query=query,
            user_groups=groups,
            top_k=TOP_K,
        )

        lexical_ms = (
            time.perf_counter() - start
        ) * 1000

        lexical_ids = add_result(
            lexical_metrics,
            expected,
            lexical,
            lexical_ms,
        )

        # ----------------------------
        # Query embedding
        # ----------------------------

        start = time.perf_counter()

        query_vector = embed_text(
            client=embedding_client,
            deployment=azure_config[
                "embedding_deployment"
            ],
            text=query,
        )

        embedding_ms = (
            time.perf_counter() - start
        ) * 1000

        embedding_latencies.append(
            embedding_ms
        )

        # ----------------------------
        # Vector
        # ----------------------------

        start = time.perf_counter()

        vector = governed_vector_search(
            client=search_client,
            query_vector=query_vector,
            user_groups=groups,
            top_k=TOP_K,
        )

        vector_ms = (
            time.perf_counter() - start
        ) * 1000

        vector_ids = add_result(
            vector_metrics,
            expected,
            vector,
            vector_ms,
        )

        # ----------------------------
        # Hybrid
        # ----------------------------

        start = time.perf_counter()

        hybrid = governed_hybrid_search(
            client=search_client,
            query=query,
            query_vector=query_vector,
            user_groups=groups,
            top_k=TOP_K,
        )

        hybrid_ms = (
            time.perf_counter() - start
        ) * 1000

        hybrid_ids = add_result(
            hybrid_metrics,
            expected,
            hybrid,
            hybrid_ms,
        )

        print(
            f"{case['id']}"
        )

        print(
            f"  Expected: {expected}"
        )

        print(
            f"  Lexical:  {lexical_ids}"
        )

        print(
            f"  Vector:   {vector_ids}"
        )

        print(
            f"  Hybrid:   {hybrid_ids}"
        )

        print()

    summaries = [

        summarize(
            "Lexical",
            lexical_metrics,
        ),

        summarize(
            "Vector",
            vector_metrics,
        ),

        summarize(
            "Hybrid",
            hybrid_metrics,
        ),
    ]

    print(
        "\n"
        "====================================\n"
        " SUMMARY\n"
        "===================================="
    )

    print(
        f"\n{'Mode':<12}"
        f"{'Recall@5':>12}"
        f"{'Precision@5':>15}"
        f"{'Hit@5':>10}"
        f"{'p95 ms':>12}"
    )

    for item in summaries:

        print(
            f"{item['mode']:<12}"
            f"{item['recall@5']:>12.3f}"
            f"{item['precision@5']:>15.3f}"
            f"{item['hit@5']:>10.3f}"
            f"{item['p95_ms']:>12.1f}"
        )

    print(
        "\nEmbedding latency p95: "
        f"{np.percentile(embedding_latencies, 95):.1f} ms"
    )


if __name__ == "__main__":
    main()