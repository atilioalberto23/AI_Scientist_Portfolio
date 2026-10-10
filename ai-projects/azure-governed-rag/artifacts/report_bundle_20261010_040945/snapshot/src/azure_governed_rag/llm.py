from openai import OpenAI

from azure_governed_rag.identity import (
    build_token_provider,
)


def build_client(base_url: str):

    token_provider = build_token_provider()

    return OpenAI(
        base_url=base_url,
        api_key=token_provider,
    )


def run_inference(
    client,
    deployment: str,
    prompt: str,
):

    response = client.responses.create(
        model=deployment,
        input=prompt,
    )

    return response.output_text