from pathlib import Path
import sys
import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
)

from azure_governed_rag.search import (
    build_index_client,
)


def load_config():

    path = LAB_ROOT / "config.yaml"

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def build_index_definition(
    index_name: str
):

    fields = [

        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),

        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False
        ),

        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False
        ),

        SimpleField(
            name="doc_id",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),

        SimpleField(
            name="version",
            type=SearchFieldDataType.String,
            filterable=True
        ),

        SimpleField(
            name="effective_from",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True
        ),

        SimpleField(
            name="effective_to",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True
        ),

        SearchableField(
            name="group_ids",
            type=SearchFieldDataType.String,
            collection=True,
            searchable=False,
            filterable=True
        ),

        SimpleField(
            name="source_url",
            type=SearchFieldDataType.String,
            filterable=False
        ),
    ]

    return SearchIndex(
        name=index_name,
        fields=fields
    )


def main():

    config = load_config()

    search_config = config["search"]

    endpoint = search_config["endpoint"]
    index_name = search_config["index_name"]

    client = build_index_client(
        endpoint
    )

    index = build_index_definition(
        index_name
    )

    result = client.create_or_update_index(
        index
    )

    print(
        f"Index ready: {result.name}"
    )


if __name__ == "__main__":
    main()