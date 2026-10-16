// API client. Same-origin when the console is served by the API (/console/); ?api=http://host:port overrides.
const params = new URLSearchParams(location.search);
export const API_BASE = (params.get("api") || (location.protocol.startsWith("http") && location.port !== "5500"
  ? location.origin : "http://127.0.0.1:8000")).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, body) {
    super(body?.message || `HTTP ${status}`);
    this.status = status;
    this.code = body?.code || "error";
    this.hint = body?.hint || "";
    this.requestId = body?.request_id || "";
  }
}

async function request(method, path, body, { timeoutMs = 300000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (e) {
    throw new ApiError(0, {
      code: e.name === "AbortError" ? "timeout" : "offline",
      message: e.name === "AbortError" ? "The request took too long and was cancelled." : `Cannot reach the API at ${API_BASE}.`,
      hint: e.name === "AbortError" ? "The first LLM answer after a cold start can take ~100 s; try again."
        : "Start it with: uvicorn src.backend.main:app",
    });
  } finally {
    clearTimeout(timer);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const api = {
  get: (path, opts) => request("GET", path, null, opts),
  post: (path, body, opts) => request("POST", path, body, opts),
};
