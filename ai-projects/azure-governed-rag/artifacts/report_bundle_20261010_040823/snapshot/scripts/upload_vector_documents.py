from pathlib import Path
import json
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
)


def load_config():

    with (LAB_ROOT / "config.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_documents():

    path = (
        LAB_ROOT
        / "data"
        / "synthetic"
        / "policies.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():

    config = load_config()

    azure_config = config["azure"]
    search_config = config["search"]

    embedding_client = build_embedding_client(
        base_url=azure_config["openai_base_url"]
    )

    search_client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"],
    )

    documents = load_documents()

    print(
        f"\nVectorizing {len(documents)} documents...\n"
    )

    for i, document in enumerate(
        documents,
        start=1,
    ):

        document["content_vector"] = embed_text(
            client=embedding_client,
            deployment=azure_config[
                "embedding_deployment"
            ],
            text=document["content"],
        )

        print(
            f"[{i}/{len(documents)}] "
            f"{document['id']}"
        )

    results = search_client.upload_documents(
        documents=documents
    )

    succeeded = sum(
        result.succeeded
        for result in results
    )

    print(
        f"\n✓ Uploaded: "
        f"{succeeded}/{len(results)}"
    )


if __name__ == "__main__":
    main()