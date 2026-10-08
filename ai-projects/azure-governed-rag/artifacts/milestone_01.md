# Milestone 01 — Keyless Azure inference

Status: PASS

Flow:
Azure CLI login
→ DefaultAzureCredential
→ Entra bearer token
→ Azure OpenAI v1 endpoint
→ gpt-5-mini deployment
→ Responses API

Expected response:
AZURE_SLICE_OK

Authentication:
Keyless / Microsoft Entra ID

Token audience:
https://cognitiveservices.azure.com/.default