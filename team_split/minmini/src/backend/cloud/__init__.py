"""Azure integration for BACKEND_MODE=azure (plan A): Key Vault secrets, Blob storage, Application Insights.

Authentication is Microsoft Entra ID through ``DefaultAzureCredential`` (``az login`` on this machine); the storage
account has shared keys disabled, so no storage key or connection string exists. Secret values are never logged,
returned by an endpoint or written to a file.
"""
