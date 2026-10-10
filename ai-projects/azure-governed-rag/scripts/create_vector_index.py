from pathlib import Path
import sys
import yaml


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
)

from azure_governed_rag.search import build_index_client


EMBEDDING_DIM = 1536   # <-- usa aquí el valor REAL de tu smoke test


def load_config():
    with (LAB_ROOT / "config.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def build_index(index_name: str):

    fields = [

        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),

        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),

        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
        ),

        SimpleField(
            name="doc_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),

        SimpleField(
            name="version",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="effective_from",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
        ),

        SimpleField(
            name="effective_to",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
        ),

        SearchField(
            name="group_ids",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.String
            ),
            filterable=True,
            retrievable=True,
        ),

        SimpleField(
            name="source_url",
            type=SearchFieldDataType.String,
        ),

        # NUEVO
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=EMBEDDING_DIM,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    metric="cosine",
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )


def main():

    config = load_config()

    endpoint = config["search"]["endpoint"]
    index_name = config["search"]["index_name"]

    client = build_index_client(endpoint)

    index = build_index(index_name)

    result = client.create_or_update_index(index)

    print(
        f"\n✓ Vector index ready: {result.name}"
    )


if __name__ == "__main__":
    main()