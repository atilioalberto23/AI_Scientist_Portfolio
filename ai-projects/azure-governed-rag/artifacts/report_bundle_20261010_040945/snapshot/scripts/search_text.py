from pathlib import Path
from datetime import datetime, timezone

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


def build_security_filter(
    user_groups,
    now
):

    group_expression = " or ".join(
        f"group_ids/any(g: g eq '{group}')"
        for group in user_groups
    )

    now_iso = (
        now
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    validity_expression = (
        f"effective_from le {now_iso} "
        "and "
        f"(effective_to eq null "
        f"or effective_to gt {now_iso})"
    )

    return (
        f"({group_expression}) "
        f"and ({validity_expression})"
    )


def main():

    config = load_config()

    search_config = config["search"]

    client = build_search_client(
        endpoint=search_config["endpoint"],
        index_name=search_config["index_name"]
    )

    query = (
        "What approval is required "
        "for a large corporate credit facility?"
    )

    user_groups = [
        "risk"
    ]

    now = datetime.now(
        timezone.utc
    )

    security_filter = (
        build_security_filter(
            user_groups=user_groups,
            now=now
        )
    )

    print(
        "\nQuery:"
    )

    print(query)

    print(
        "\nSecurity filter:"
    )

    print(
        security_filter
    )

    results = client.search(
        search_text=query,
        filter=security_filter,
        select=[
            "id",
            "title",
            "content",
            "doc_id",
            "page",
            "version",
            "group_ids"
        ],
        top=5
    )

    print(
        "\nAuthorized results:\n"
    )

    count = 0

    for result in results:

        count += 1

        print(
            f"[{count}] "
            f"{result['title']}"
        )

        print(
            f"    page: "
            f"{result['page']}"
        )

        print(
            f"    groups: "
            f"{result['group_ids']}"
        )

        print(
            f"    content: "
            f"{result['content']}"
        )

        print()

    if count == 0:

        print(
            "No authorized evidence found."
        )


if __name__ == "__main__":
    main()