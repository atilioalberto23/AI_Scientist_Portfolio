from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)


AZURE_AI_SCOPE = (
    "https://cognitiveservices.azure.com/.default"
)


def build_credential():
    return DefaultAzureCredential()


def build_token_provider():

    credential = build_credential()

    return get_bearer_token_provider(
        credential,
        AZURE_AI_SCOPE,
    )