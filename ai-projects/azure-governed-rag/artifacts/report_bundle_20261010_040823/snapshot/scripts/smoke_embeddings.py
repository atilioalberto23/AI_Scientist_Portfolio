from pathlib import Path
import sys

import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from azure_governed_rag.embeddings import (
    build_embedding_client,
    embed_text,
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

    print(
        "\n"
        "====================================\n"
        " AZURE GOVERNED RAG · EMBEDDING TEST\n"
        "====================================\n"
    )

    config = load_config()

    azure_config = config["azure"]

    client = build_embedding_client(
        base_url=azure_config["openai_base_url"]
    )

    text = (
        "Corporate credit facilities above "
        "USD 5 million require Risk Committee approval."
    )

    vector = embed_text(
        client=client,
        deployment=azure_config[
            "embedding_deployment"
        ],
        text=text,
    )

    print("Input:")
    print(text)

    print(
        f"\nEmbedding dimensions: {len(vector)}"
    )

    print(
        "\nFirst 10 values:"
    )

    print(
        vector[:10]
    )

    print(
        "\n✓ Text → embedding path verified."
    )


if __name__ == "__main__":
    main()