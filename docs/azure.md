# Azure mode (plan A)

Authorised by the user on 2026-10-16. Local mode stays the default and the evaluated stack (C1-C6).
Nothing in this file is a secret: it lists resource names and endpoints only.

## What runs where

| part | where |
|---|---|
| FastAPI app, console, forecast models, fault diagnosis, decision engine | local |
| LLM (Ollama, qwen2.5:7b-instruct), MiniLM embeddings, cross-encoder | local |
| Secure pipeline index (BM25 + MiniLM + rerank, in memory) | local (unchanged, so the C3-C6 results still apply) |
| Vector store for `BACKEND_MODE=azure` (`/ingest`, `/query`, `/query/top`, `/ask`) | Azure AI Search Free, one index `greenrag`, domains as a filter |
| Knowledge base `documents.jsonl` (`KB_SOURCE=blob`) and `results/` | Azure Blob Storage, containers `kbdocs` and `results` |
| Secrets (Search admin key, App Insights connection string) | Azure Key Vault |
| Request telemetry (`GREENRAG_APPINSIGHTS=true`) | Application Insights on Log Analytics (0.1 GB/day cap) |

## Resources (resource group `greenragsec-rg`, centralindia)

| resource | name | SKU |
|---|---|---|
| Storage account | `greenragsecymjvr` | Standard_LRS, Hot, shared keys off, no public access, TLS 1.2 |
| Key Vault | `greenragsec-kv-ymjvr` | standard, RBAC, 7-day soft delete |
| Log Analytics | `greenragsec-law-ymjvr` | PerGB2018, 30 days, 0.1 GB/day cap |
| Application Insights | `greenragsec-appi-ymjvr` | workspace-based |
| AI Search | `greenragsec-srch-ymjvr` | free (50 MB, 3 indexes) |

The resources were created with the az commands in PROGRESS.md (Azure stage 1). The cost estimate (about $0.01 per
month) is in the same section. The user's budget alert is $20/month.

## Configuration

Put these in your local, git-ignored `.env`; `.env.example` lists every name.

```
BACKEND_MODE=azure
AZURE_SEARCH_ENDPOINT=https://greenragsec-srch-ymjvr.search.windows.net
AZURE_SEARCH_INDEX=greenrag
AZURE_KEY_VAULT_URL=https://greenragsec-kv-ymjvr.vault.azure.net/
AZURE_STORAGE_ACCOUNT_URL=https://greenragsecymjvr.blob.core.windows.net
KB_SOURCE=blob                  # optional
GREENRAG_APPINSIGHTS=true       # optional
```

How secrets are handled:
- **Never in a file.** `AZURE_SEARCH_API_KEY` and `APPLICATIONINSIGHTS_CONNECTION_STRING` stay empty; the app reads
  them from Key Vault (`src/backend/cloud/secrets.py`).
- **Entra login, no keys.** The app signs in with `DefaultAzureCredential` (your `az login`), using the roles Key
  Vault Secrets Officer and Storage Blob Data Contributor. The storage account has no keys enabled, so no
  connection string exists.
- **Names only in logs and APIs.** Secret values are never logged or returned; `/system/info` shows host names only.

## Commands

```powershell
python -m src.backend.cloud.blob upload kb          # data/processed/knowledge_base -> kbdocs
python -m src.backend.cloud.blob upload results     # results/ -> results
python -m src.backend.cloud.blob list results
$env:BACKEND_MODE='azure'; uvicorn src.backend.main:app   # with the .env above
```

`POST /ingest/{domain}` in azure mode embeds locally and uploads to the shared index.

## Status

- **Code: done** (Azure stage 2).
- **Tests: mocked only** (`tests/test_azure_mode.py`). No test calls Azure.
- **I4 local-vs-Azure comparison: not run yet** (stage 3); `results/cloud/i4_status.md` has no Azure numbers until it
  runs.
