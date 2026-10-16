# I4 cloud portability: local vs Azure (plan A)

Measured 2026-10-16T14:58:47Z with `python -m experiments.i4_harness` (reduced item set, user request).
Both modes ran on this machine through the same app (in-process TestClient). Azure mode used AI Search Free
(index `greenrag`), Blob (`kbdocs`), Key Vault and App Insights in centralindia. MiniLM embeddings and the
Ollama LLM (qwen2.5:7b-instruct) are local in both modes, so latency differences come from the Azure calls.

Warm-up before measuring (one each, excluded from the latencies): local 71.8 s, Azure 72.5 s.

## Latency per endpoint (ms)

/query and /forecast: all requests of 3 repetitions after warm-up. LLM endpoints: one run; median and p95
across the items of that run.

| endpoint | n local (ok) | local median | local p95 | n Azure (ok) | Azure median | Azure p95 |
|---|---|---|---|---|---|---|
| query | 192 (192) | 36 | 41 | 192 (192) | 456 | 523 |
| ask | 16 (16) | 1,919 | 5,280 | 16 (16) | 3,496 | 5,419 |
| secure_ask | 10 (10) | 1,088 | 2,325 | 10 (10) | 1,062 | 2,286 |
| decision | 10 (10) | 935 | 2,255 | 10 (10) | 945 | 2,262 |
| diagnose | 5 (5) | 7,552 | 7,873 | 5 (5) | 7,519 | 7,873 |
| forecast | 150 (150) | 27 | 50 | 150 (150) | 23 | 46 |

## /query retrieval equivalence (knowledge_base domain, top 5, first repetition)

- Same top-5 chunks in the same order: 64/64 queries.
- Same top-5 set: 64/64; mean Jaccard overlap 1.000 (n = 64).
- nDCG@5 (document level, B2 labels): local 0.614, Azure 0.614 (n = 64).

## Output equivalence (first run; exact match = every compared field identical)

| endpoint | items compared | exact match | answer text same | sources/evidence same | other |
|---|---|---|---|---|---|
| query | 64 | 64/64 | - | - | - |
| ask | 16 | 6/16 | 6/16 | 16/16 | - |
| secure_ask | 10 | 10/10 | 10/10 | 10/10 | - |
| decision | 10 | 10/10 | 10/10 | 10/10 | same decision 10/10 |
| diagnose | 5 | 5/5 | - | - | same component 5/5, same evidence 5/5 |
| forecast | 50 | 50/50 | - | - | - |

Notes:
- /ask: the legacy generator calls Ollama with default sampling (no temperature or seed), so its answer text
  varies between any two runs; the retrieved sources were the same for every item. /secure-ask, /decision
  and /diagnose use temperature 0 and seed 0.
- /secure-ask and /decision include 5 query-injection items; they were blocked and rejected in both modes,
  so those 5 items match without reaching the LLM.
- /query median rises from 36 to 456 ms (n = 192 each): each Azure request calls AI
  Search over the internet and, because the legacy endpoints build a new store per request, also lists
  the indexes first. The /ask difference is not separable from the LLM variance in this sample.

## Not run

- /ask on the other 48 labelled queries, /secure-ask and /decision on the other 54, /diagnose on the other
  17 Farm A events (reduced plan).
- No repetitions for the LLM endpoints (one run each), so their latency includes Ollama variance.
- No KB-poisoning attacks in Azure mode (they need an attacked store); the 5 attack items are query injections.
- No hosted API: the app runs locally in both modes; user-to-cloud network latency is not measured.
- No cost measurement here.
