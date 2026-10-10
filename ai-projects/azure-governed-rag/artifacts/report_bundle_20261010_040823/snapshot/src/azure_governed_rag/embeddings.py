from openai import OpenAI

from azure_governed_rag.identity import (
    build_token_provider,
)


def build_embedding_client(
    base_url: str,
):

    return OpenAI(
        base_url=base_url,
        api_key=build_token_provider(),
    )


def embed_text(
    client,
    deployment: str,
    text: str,
):

    response = client.embeddings.create(
        model=deployment,
        input=text,
    )

    return response.data[0].embedding