from azure.identity import DefaultAzureCredential

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient


SEARCH_AUDIENCE = "https://search.azure.com"


def build_search_credential():

    return DefaultAzureCredential()


def build_index_client(
    endpoint: str
):

    credential = build_search_credential()

    return SearchIndexClient(
        endpoint=endpoint,
        credential=credential,
        audience=SEARCH_AUDIENCE
    )


def build_search_client(
    endpoint: str,
    index_name: str
):

    credential = build_search_credential()

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=credential,
        audience=SEARCH_AUDIENCE
    )



from datetime import datetime, timezone

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient


SEARCH_AUDIENCE = "https://search.azure.com"


def build_search_credential():
    return DefaultAzureCredential()


def build_index_client(endpoint: str):

    return SearchIndexClient(
        endpoint=endpoint,
        credential=build_search_credential(),
        audience=SEARCH_AUDIENCE,
    )


def build_search_client(
    endpoint: str,
    index_name: str,
):

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=build_search_credential(),
        audience=SEARCH_AUDIENCE,
    )


def build_security_filter(
    user_groups,
    now=None,
):

    if not user_groups:
        raise ValueError(
            "At least one user group is required."
        )

    if now is None:
        now = datetime.now(
            timezone.utc
        )

    group_expression = " or ".join(
        f"group_ids/any(g: g eq '{group}')"
        for group in user_groups
    )

    now_iso = (
        now.astimezone(timezone.utc)
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


def governed_search(
    client,
    query,
    user_groups,
    top_k=5,
    now=None,
):

    security_filter = (
        build_security_filter(
            user_groups=user_groups,
            now=now,
        )
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
            "effective_from",
            "effective_to",
            "group_ids",
            "source_url",
        ],
        top=top_k,
    )

    return list(results)

from azure.search.documents.models import (
    VectorizedQuery,
)


def governed_vector_search(
    client,
    query_vector,
    user_groups,
    top_k=5,
    now=None,
):

    security_filter = build_security_filter(
        user_groups=user_groups,
        now=now,
    )

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
        kind="vector",
    )

    results = client.search(
        search_text=None,

        vector_queries=[
            vector_query
        ],

        filter=security_filter,

        vector_filter_mode="preFilter",

        select=[
            "id",
            "title",
            "content",
            "doc_id",
            "page",
            "version",
            "group_ids",
            "source_url",
        ],

        top=top_k,
    )

    return list(results)


def governed_lexical_search(
    client,
    query,
    user_groups,
    top_k=5,
    now=None,
):

    security_filter = build_security_filter(
        user_groups=user_groups,
        now=now,
    )

    results = client.search(
        search_text=query,
        search_fields=[
            "title",
            "content",
        ],
        filter=security_filter,
        select=[
            "id",
            "doc_id",
            "title",
            "content",
            "page",
            "version",
            "group_ids",
            "source_url",
        ],
        top=top_k,
    )

    return list(results)

def governed_hybrid_search(
    client,
    query,
    query_vector,
    user_groups,
    top_k=5,
    candidate_k=10,
    now=None,
):

    security_filter = build_security_filter(
        user_groups=user_groups,
        now=now,
    )

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=candidate_k,
        fields="content_vector",
        kind="vector",
    )

    results = client.search(
        search_text=query,

        vector_queries=[
            vector_query
        ],

        vector_filter_mode="preFilter",

        filter=security_filter,

        search_fields=[
            "title",
            "content",
        ],

        select=[
            "id",
            "doc_id",
            "title",
            "content",
            "page",
            "version",
            "group_ids",
            "source_url",
        ],

        top=top_k,
    )

    return list(results)