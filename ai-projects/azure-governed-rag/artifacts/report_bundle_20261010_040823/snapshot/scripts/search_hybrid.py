from pathlib import Path
import sys
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
    governed_hybrid_search,
)


def load_config():

    with (LAB_ROOT / "config.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def main():

    config = load_config()

    azure_config = config["azure"]
    search_config = config["search"]

    query = (
        "Who needs to authorize "
        "a large business loan?"
    )

    groups = [
        "risk"
    ]

    embedding_client = build_embedding_client(
        base_url=azure_config[
            "openai_base_url"
        ]
    )

    query_vector = embed_text(
        client=embedding_client,
        deployment=azure_config[
            "embedding_deployment"
        ],
        text=query,
    )

    search_client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"],
    )

    results = governed_hybrid_search(
        client=search_client,
        query=query,
        query_vector=query_vector,
        user_groups=groups,
        top_k=5,
    )

    print(
        "\n"
        "====================================\n"
        " HYBRID SEARCH\n"
        "====================================\n"
    )

    print(f"Query: {query}")
    print(f"Groups: {groups}\n")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"[{rank}] "
            f"{result['title']}"
        )

        print(
            f"    doc_id: "
            f"{result['doc_id']}"
        )

        print(
            f"    score: "
            f"{result['@search.score']:.6f}"
        )

        print(
            f"    {result['content']}\n"
        )


if __name__ == "__main__":
    main()