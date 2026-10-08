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