from pathlib import Path
import json
import sys
import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from azure_governed_rag.search import (
    build_search_client,
)


def load_config():

    with (
        LAB_ROOT / "config.yaml"
    ).open(
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

    search_config = config["search"]

    client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"]
    )

    documents = load_documents()

    result = client.upload_documents(
        documents=documents
    )

    succeeded = sum(
        item.succeeded
        for item in result
    )

    print(
        f"Uploaded successfully: "
        f"{succeeded}/{len(result)}"
    )


if __name__ == "__main__":
    main()