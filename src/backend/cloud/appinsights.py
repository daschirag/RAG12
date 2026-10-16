"""Send the structured request log (greenrag.access) to Application Insights.

A small logging handler posts each request line as an Application Insights "Request" item to the documented
ingestion endpoint (``<IngestionEndpoint>/v2/track``). It uses httpx, which is already installed, instead of the
OpenTelemetry exporter: that exporter would change the opentelemetry versions chromadb depends on.
Enabled only with GREENRAG_APPINSIGHTS=true; the connection string comes from the environment or Key Vault and is
never logged. Items are batched and sent from a background thread; a failed send is dropped, never raised.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timedelta, timezone

import httpx

from ..settings import get_settings

ACCESS_LOGGER = "greenrag.access"
_installed: "AppInsightsHandler | None" = None


def parse_connection_string(cs: str) -> tuple[str, str]:
    """(instrumentation key, ingestion endpoint); raises ValueError without echoing the string."""
    parts = dict(p.split("=", 1) for p in cs.split(";") if "=" in p)
    ikey, endpoint = parts.get("InstrumentationKey", ""), parts.get("IngestionEndpoint", "")
    if not ikey or not endpoint:
        raise ValueError("App Insights connection string lacks InstrumentationKey or IngestionEndpoint")
    return ikey, endpoint.rstrip("/")


def request_item(record: dict, ikey: str) -> dict:
    """One Application Insights Request envelope from a greenrag.access JSON line."""
    ms = float(record.get("latency_ms", 0.0))
    now = datetime.now(timezone.utc)
    d = timedelta(milliseconds=ms)
    hours, rem = divmod(d.total_seconds(), 3600)
    minutes, seconds = divmod(rem, 60)
    status = int(record.get("status", 0))
    return {
        "name": "Microsoft.ApplicationInsights.Request", "time": (now - d).isoformat(), "iKey": ikey,
        "tags": {"ai.cloud.role": "greenrag-sec-api", "ai.operation.id": record.get("request_id", ""),
                 "ai.operation.name": f"{record.get('method', '')} {record.get('path', '')}"},
        "data": {"baseType": "RequestData", "baseData": {
            "ver": 2, "id": record.get("request_id", ""), "name": f"{record.get('method', '')} {record.get('path', '')}",
            "duration": f"{int(d.days):d}.{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}",
            "responseCode": str(status), "success": status < 500, "url": record.get("path", "")}},
    }


class AppInsightsHandler(logging.Handler):
    def __init__(self, connection_string: str, client: httpx.Client | None = None, batch: int = 20,
                 flush_s: float = 5.0) -> None:
        super().__init__(level=logging.INFO)
        self.ikey, endpoint = parse_connection_string(connection_string)
        self.url = f"{endpoint}/v2/track"
        self.client = client or httpx.Client(timeout=10)
        self.batch, self.flush_s = batch, flush_s
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=10_000)
        self.sent = self.failed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="appinsights", daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = json.loads(record.getMessage())
            if data.get("event") == "request":
                self.q.put_nowait(request_item(data, self.ikey))
        except (ValueError, queue.Full):
            pass

    def flush(self) -> None:
        items = []
        while not self.q.empty() and len(items) < 500:
            items.append(self.q.get_nowait())
        if not items:
            return
        try:
            r = self.client.post(self.url, content="\n".join(json.dumps(i) for i in items),
                                 headers={"Content-Type": "application/x-json-stream"})
            if r.status_code < 300:
                self.sent += len(items)
            else:
                self.failed += len(items)
        except httpx.HTTPError:
            self.failed += len(items)

    def _loop(self) -> None:
        while not self._stop.wait(self.flush_s):
            self.flush()

    def close(self) -> None:
        self._stop.set()
        self.flush()
        super().close()


def install(connection_string: str | None = None, client: httpx.Client | None = None) -> "AppInsightsHandler | None":
    """Attach the handler to greenrag.access if GREENRAG_APPINSIGHTS=true (idempotent). Returns it or None."""
    global _installed
    if not get_settings().appinsights_enabled:
        return None
    if _installed is not None:
        return _installed
    if connection_string is None:
        from .secrets import get_secret

        connection_string = get_secret("appinsights_connection_string")
    _installed = AppInsightsHandler(connection_string, client=client)
    logging.getLogger(ACCESS_LOGGER).addHandler(_installed)
    return _installed


def uninstall() -> None:
    global _installed
    if _installed is not None:
        logging.getLogger(ACCESS_LOGGER).removeHandler(_installed)
        _installed.close()
        _installed = None
