"""Blob storage for the knowledge base and results (Entra auth; the account has shared keys disabled).

    python -m src.backend.cloud.blob upload kb        # data/processed/knowledge_base -> container kbdocs
    python -m src.backend.cloud.blob upload results   # results/ -> container results
    python -m src.backend.cloud.blob download kb      # container kbdocs -> data/processed/blob_cache/kb
    python -m src.backend.cloud.blob list results

With KB_SOURCE=blob the secure pipeline reads documents.jsonl from the kb container (downloaded to the git-ignored
cache). The trust check is unchanged: every document is still verified against the in-process registry built
from code and datasets, so a tampered blob is rejected (fail closed).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable

from ..settings import PROJECT_ROOT, get_settings

LOCAL_DIRS = {"kb": PROJECT_ROOT / "data" / "processed" / "knowledge_base", "results": PROJECT_ROOT / "results"}
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "blob_cache"


def _service_client(account_url: str):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient(account_url=account_url,
                             credential=DefaultAzureCredential(exclude_interactive_browser_credential=True))


# Tests replace this factory with an in-memory fake.
service_factory: Callable[[str], object] = _service_client


def container_name(which: str) -> str:
    s = get_settings()
    return {"kb": s.azure_kb_container, "results": s.azure_results_container}[which]


def _container(which: str):
    url = get_settings().azure_storage_account_url
    if not url:
        raise RuntimeError("Missing required Azure configuration: AZURE_STORAGE_ACCOUNT_URL is unset "
                           "(e.g. https://<account>.blob.core.windows.net).")
    return service_factory(url).get_container_client(container_name(which))


def upload_dir(which: str, source: Path | None = None) -> int:
    """Upload every file under the local folder (relative paths as blob names); returns the file count."""
    src = source or LOCAL_DIRS[which]
    cc = _container(which)
    n = 0
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        with f.open("rb") as fh:
            cc.upload_blob(name=f.relative_to(src).as_posix(), data=fh, overwrite=True)
        n += 1
    return n


def download_dir(which: str, dest: Path | None = None) -> int:
    out = dest or CACHE_DIR / which
    cc = _container(which)
    n = 0
    for b in cc.list_blobs():
        target = out / b.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cc.download_blob(b.name).readall())
        n += 1
    return n


def kb_documents_path() -> Path:
    """documents.jsonl for the secure pipeline: the local build, or (KB_SOURCE=blob) the copy from the kb container."""
    if get_settings().kb_source != "blob":
        return LOCAL_DIRS["kb"] / "documents.jsonl"
    target = CACHE_DIR / "kb" / "documents.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_container("kb").download_blob("documents.jsonl").readall())
    return target


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("upload", "download", "list"))
    ap.add_argument("which", choices=("kb", "results"))
    args = ap.parse_args()
    if args.action == "upload":
        print(f"uploaded {upload_dir(args.which)} file(s) to container {container_name(args.which)}")
    elif args.action == "download":
        print(f"downloaded {download_dir(args.which)} blob(s) to {CACHE_DIR / args.which}")
    else:
        names = [b.name for b in _container(args.which).list_blobs()]
        print(f"{len(names)} blob(s) in {container_name(args.which)}")


if __name__ == "__main__":
    main()
