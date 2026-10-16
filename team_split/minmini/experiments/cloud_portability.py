"""I4 - Cloud portability: is the Azure configuration usable from here? (never simulated)

    python -m experiments.cloud_portability   -> results/cloud/i4_status.{json,md}

Checks only whether the settings the BACKEND_MODE=azure code path reads are present (names, never values) and
whether the Azure CLI is installed. Nothing is sent to Azure. If any required setting is missing, the local-vs-Azure
equivalence and latency harness is PENDING and no Azure number is reported. Azure plan A (AI Search Free, Blob,
Key Vault, Log Analytics + App Insights; MiniLM and Ollama local) was authorised on 2026-10-16.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cloud"
# Plan A settings (src/backend/settings.py, .env.example). The Search admin key and the App Insights connection
# string are read from Key Vault, so they are not required here; Azure OpenAI is not used by plan A.
REQUIRED = ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_INDEX", "AZURE_KEY_VAULT_URL", "AZURE_STORAGE_ACCOUNT_URL")


def check(env: dict[str, str | None], env_file: Path) -> dict:
    present = {k: bool((env.get(k) or "").strip()) for k in REQUIRED}
    missing = [k for k, ok in present.items() if not ok]
    reasons = []
    if not env_file.is_file():
        reasons.append(".env file not present")
    if missing:
        reasons.append("required Azure settings not set: " + ", ".join(missing))
    if shutil.which("az") is None:
        reasons.append("Azure CLI (az) not installed (informational; the code path does not need it)")
    reasons.append("Azure plan A authorised by the user on 2026-10-16 (CLAUDE.md rule 12); the I4 harness runs only "
                   "with these settings present")
    return {"usable": not missing, "settings_present": present, "env_file": env_file.is_file(),
            "status": "READY" if not missing else "PENDING", "reasons": reasons}


def main() -> None:
    env_file = ROOT / ".env"
    env = {**(dotenv_values(env_file) if env_file.is_file() else {}), **os.environ}
    res = check(env, env_file)
    res["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "i4_status.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    L = ["# I4 cloud portability status", "", f"Checked {res['timestamp_utc']} by `python -m experiments.cloud_portability`.",
         "", f"**Status: {res['status']}.** Local-vs-Azure functional equivalence and latency: "
         + ("not run yet (configuration present; needs the user's go-ahead under CLAUDE.md rule 12)."
            if res["usable"] else "PENDING, not measured and not simulated."), "", "Reasons:", ""]
    L += [f"- {r}" for r in res["reasons"]]
    L += ["", "| setting | present |", "|---|---|"] + [f"| {k} | {'yes' if v else 'no'} |"
                                                      for k, v in res["settings_present"].items()]
    L += ["", "No Azure latency, cost or equivalence number exists in this repo."]
    (OUT / "i4_status.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"I4 {res['status']}: " + "; ".join(res["reasons"]))


if __name__ == "__main__":
    main()
