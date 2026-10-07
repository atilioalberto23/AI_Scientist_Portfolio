from azure.identity import DefaultAzureCredential


credential = DefaultAzureCredential()

token = credential.get_token(
    "https://management.azure.com/.default"
)

print("Azure authentication OK")
print("Token expires at:", token.expires_on)