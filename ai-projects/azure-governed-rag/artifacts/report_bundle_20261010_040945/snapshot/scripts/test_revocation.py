from pathlib import Path

import sys
import time
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


def main():

    config = load_config()

    search_config = config["search"]

    client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"],
    )

    query = (
        "corporate credit facilities "
        "above USD 5 million"
    )

    print(
        "\nBefore revocation:"
    )

    before = governed_search(
        client=client,
        query=query,
        user_groups=["risk"],
    )

    print(
        [
            result["doc_id"]
            for result in before
        ]
    )

    client.merge_documents(
        documents=[
            {
                "id": "POL001-P01-C01",
                "group_ids": [
                    "credit"
                ],
            }
        ]
    )

    # Give the index a short moment
    # to reflect the update.
    time.sleep(2)

    print(
        "\nAfter revocation:"
    )

    after = governed_search(
        client=client,
        query=query,
        user_groups=["risk"],
    )

    after_ids = [
        result["doc_id"]
        for result in after
    ]

    print(
        after_ids
    )

    if "POL001" in after_ids:

        raise RuntimeError(
            "SECURITY FAILURE: "
            "revoked document is still visible."
        )

    print(
        "\n✓ Revocation propagated successfully."
    )


if __name__ == "__main__":
    main()