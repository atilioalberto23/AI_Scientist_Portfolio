from pathlib import Path

import sys
import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from azure_governed_rag.pipeline import (
    answer_query,
)


def load_config():

    with (
        LAB_ROOT / "config.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def main():

    config = load_config()

    question = (
        "Who must authorize "
        "a large corporate loan?"
    )

    user_groups = [
        "risk"
    ]

    result = answer_query(
        config=config,
        question=question,
        user_groups=user_groups,
        top_k=5,
    )

    print(
        "\n"
        "====================================\n"
        " AZURE GOVERNED RAG\n"
        "====================================\n"
    )

    print(
        f"Question:\n{question}\n"
    )

    print(
        f"Groups:\n{user_groups}\n"
    )

    print(
        "Answer:\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nSources:"
    )

    for source in result["sources"]:

        print(
            f"- {source['doc_id']} "
            f"p.{source['page']} "
            f"({source['version']})"
        )

    print(
        f"\nRetrieved: "
        f"{result['retrieval_count']}"
    )

    print(
        f"LLM called: "
        f"{result['llm_called']}"
    )


if __name__ == "__main__":
    main()