"""I3 - Experiments 1-7 (+ ablation): status of every step, reusing finished results, never re-running them.

    python -m experiments.run_all                 # status only: writes results/run_all_status.{md,json}
    python -m experiments.run_all --run-missing   # also runs the incomplete steps, one at a time

A step is *complete* when its declared outputs exist (and, for resumable LLM runs, hold every expected item).
Complete steps are never re-run; ``--run-missing`` starts only incomplete steps, in order, as separate
processes (one heavy job at a time) and stops before a step if free RAM is below ``--min-free-mb``.
Steps marked ``manual`` are not started automatically (they need a decision or an external resource); the
status says why. The status file links every experiment to the summary that holds its numbers.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _lines(path: Path) -> int:
    return sum(1 for x in path.read_text(encoding="utf-8").splitlines() if x.strip()) if path.is_file() else 0


def _unique(path: Path, *keys: str) -> int:
    if not path.is_file():
        return 0
    seen = set()
    for x in path.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            seen.add(tuple(r.get(k) for k in keys))
    return len(seen)


def _has_key(path: Path, key: str) -> bool:
    return path.is_file() and key in json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Step:
    name: str
    module: str
    args: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()                     # paths relative to results/
    check: Callable[[], tuple[bool, str]] | None = None
    manual: str = ""                                  # reason the step is not started automatically
    heavy: bool = False                               # needs Ollama (LLM)

    def status(self) -> tuple[bool, str]:
        missing = [o for o in self.outputs if not (RES / o).exists()]
        if missing:
            return False, "missing " + ", ".join(missing)
        if self.check is not None:
            return self.check()
        return True, "outputs present"

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]


@dataclass
class Experiment:
    number: str
    title: str
    summaries: tuple[str, ...]
    steps: list[Step] = field(default_factory=list)


def _count(path: str, expected: int, *keys: str) -> Callable[[], tuple[bool, str]]:
    def check():
        n = _unique(RES / path, *keys) if keys else _lines(RES / path)
        return n >= expected, f"{n}/{expected} items"
    return check


def _run_key(path: str, key: str) -> Callable[[], tuple[bool, str]]:
    return lambda: (_has_key(RES / path, key), f"run record '{key}' in {path}" +
                    ("" if _has_key(RES / path, key) else " missing"))


C6_HELDOUT = 15 * 66          # 15 held-out phrasings x 66 questions (64 KB + 2 C2 target questions)

EXPERIMENTS = [
    Experiment("1", "Forecasting (solar A1-A6, wind T1)",
               ("forecasting/a5_summary.md", "forecasting/wind/wind_summary.md"), [
                   Step("A1 preprocessing", "experiments.preprocess_solar",
                        outputs=("forecasting/preprocessing_summary.md",)),
                   Step("A2 features", "experiments.features_solar", outputs=("forecasting/features_summary.md",)),
                   Step("A3 models", "experiments.train_forecasters", outputs=("forecasting/a3_summary.md",)),
                   Step("A4 uncertainty", "experiments.uncertainty_solar", outputs=("forecasting/a4_summary.md",)),
                   Step("A5 rolling origin", "experiments.rolling_origin",
                        outputs=("forecasting/a5_folds/plant2_2020-06-17.csv",)),
                   Step("A5 evaluation", "experiments.evaluate_forecasting", outputs=("forecasting/a5_summary.md",)),
                   Step("Wind rolling origin", "experiments.wind_rolling",
                        outputs=("forecasting/wind/point_metrics.csv",)),
                   Step("Wind evaluation", "experiments.wind_evaluate", outputs=("forecasting/wind/wind_summary.md",)),
               ]),
    Experiment("2", "Retrieval (B2 chunking, B3 retrievers)",
               ("kb/b2_chunking_summary.md", "kb/b3_retrieval_summary.md"), [
                   Step("B2 chunking comparison", "experiments.chunking_comparison",
                        outputs=("kb/b2_chunking_summary.md",)),
                   Step("B3 retrieval comparison", "experiments.retrieval_comparison",
                        outputs=("kb/b3_retrieval_summary.md",)),
               ]),
    Experiment("3", "Fault diagnosis (B4 detectors, B5 diagnosis -> RAG)",
               ("faultdiag/b4_summary.md", "faultdiag/b4_if_threshold.md", "faultdiag/b5_summary.md"), [
                   Step("B4 fault diagnosis", "experiments.fault_diagnosis", outputs=("faultdiag/b4_summary.md",)),
                   Step("B4 IF threshold", "experiments.fault_threshold", outputs=("faultdiag/b4_if_threshold.md",)),
                   Step("B5 fault RAG", "experiments.fault_rag", outputs=("faultdiag/b5_summary.md",), heavy=True),
               ]),
    Experiment("4", "Hallucination (B6 RAG evaluation)",
               ("rag/b6_summary.md", "rag/run2/b6_run2_summary.md", "rag/run2/b6_followup_summary.md"), [
                   Step("B6 run 1", "experiments.rag_evaluation", outputs=("rag/b6_summary.md",), heavy=True),
                   Step("B6 run 2", "experiments.rag_evaluation", ("--run2",),
                        outputs=("rag/run2/b6_run2_summary.md",), heavy=True),
                   Step("B6 follow-up", "experiments.b6_followup", outputs=("rag/run2/b6_followup_summary.md",),
                        heavy=True),
               ]),
    Experiment("5", "Cyberattack on telemetry (C2 FDI/spoofing, C3 detector, C6 telemetry)",
               ("security/c2_summary.md", "security/c3_summary.md", "security/c6_summary.md"), [
                   Step("C2 telemetry attacks", "experiments.attack_simulation", ("--part", "telemetry"),
                        outputs=("security/c2_forecast.csv", "security/c2_fault_detection.csv")),
                   Step("C3 telemetry detector", "experiments.detector_evaluation", ("--part", "telemetry"),
                        outputs=("security/c3_telemetry_days.csv",)),
                   Step("C6 telemetry re-run (CPU)", "experiments.security_evaluation", ("--part", "telemetry"),
                        outputs=("security/c6_telemetry_slots.csv",)),
               ]),
    Experiment("6", "KB poisoning (C2 knowledge, C3 trust/filter, C4 secure pipeline, C5 engine, C6)",
               ("security/c2_summary.md", "security/c3_summary.md", "security/c4_summary.md", "trust/c5_summary.md",
                "security/c6_summary.md"), [
                   Step("C2 knowledge, qwen", "experiments.attack_simulation", ("--part", "knowledge"),
                        outputs=("security/c2_knowledge_raw.jsonl",),
                        check=_count("security/c2_knowledge_raw.jsonl", 594, "state", "system", "item_id"),
                        heavy=True),
                   Step("C2 knowledge, llama", "experiments.attack_simulation",
                        ("--part", "knowledge", "--model", "llama3.2:3b"),
                        outputs=("security/c2_knowledge_raw_llama3.2-3b.jsonl",),
                        check=_count("security/c2_knowledge_raw_llama3.2-3b.jsonl", 594, "state", "system", "item_id"),
                        heavy=True),
                   Step("C3 text + trust", "experiments.detector_evaluation", ("--part", "text"),
                        outputs=("security/c3_injection.csv", "security/c3_trust_docs.csv")),
                   Step("C3 evidence filter", "experiments.detector_evaluation", ("--part", "filter"),
                        outputs=("security/c3_filter_items.csv",)),
                   Step("C4 secure pipeline, llama", "experiments.secure_pipeline_eval", ("--model", "llama3.2:3b"),
                        outputs=("security/c4_raw_llama3.2-3b.jsonl",),
                        check=_count("security/c4_raw_llama3.2-3b.jsonl", 330), heavy=True),
                   Step("C5 decision engine (llama capture)", "experiments.decision_engine_eval",
                        ("--llm", "llama3.2:3b"), outputs=("trust/c5_summary.md",),
                        check=_count("trust/c5_raw_llama3.2-3b.jsonl", 330), heavy=True),
                   Step("C6 capture, qwen", "experiments.security_evaluation", ("--part", "capture"),
                        outputs=("security/c6_raw_qwen2.5-7b-instruct.jsonl",),
                        check=_count("security/c6_raw_qwen2.5-7b-instruct.jsonl", 330, "state", "system", "item_id"),
                        heavy=True),
                   Step("C6 qwen judge", "experiments.security_evaluation", ("--part", "judge"),
                        outputs=("security/c6_judgements.jsonl",), check=_run_key("security/c6_run.json", "judge"),
                        heavy=True),
               ]),
    Experiment("7", "Prompt injection (C2 query/KB injection, C3 detector, C6 held-out phrasings)",
               ("security/c2_summary.md", "security/c3_summary.md", "security/c6_summary.md"), [
                   Step("C6 held-out injections, qwen", "experiments.security_evaluation", ("--part", "heldout"),
                        outputs=("security/c6_heldout_raw_qwen2.5-7b-instruct.jsonl",),
                        check=_count("security/c6_heldout_raw_qwen2.5-7b-instruct.jsonl", C6_HELDOUT,
                                     "phrasing", "item_id"), heavy=True),
                   Step("C6 report", "experiments.security_evaluation", ("--report",),
                        outputs=("security/c6_summary.md",)),
               ]),
    Experiment("A-F", "Ablation configs A-F (orchestrator, src/backend/orchestrator.py)", (), [
        Step("Ablation A-F", "experiments.ablation", outputs=("ablation/ablation_summary.md",),
             manual="no ablation experiment has been designed or run: needs an LLM run per config (A-F) on a "
                    "fixed item set; not started without a decision on items and budget"),
    ]),
]


def free_ram_mb() -> float:
    """Available physical memory (Windows GlobalMemoryStatusEx); inf where unavailable."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / 2 ** 20
    except (AttributeError, OSError):
        return float("inf")


def collect() -> list[dict]:
    rows = []
    for e in EXPERIMENTS:
        for s in e.steps:
            done, detail = s.status()
            state = "complete" if done else ("manual" if s.manual else "incomplete")
            rows.append({"experiment": e.number, "title": e.title, "step": s.name, "state": state,
                         "detail": s.manual if (s.manual and not done) else detail, "heavy": s.heavy,
                         "command": " ".join(["python", "-m", s.module, *s.args])})
    return rows


def write_status(rows: list[dict], log: list[dict]) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RES / "run_all_status.json").write_text(json.dumps({"timestamp_utc": stamp, "steps": rows, "this_run": log},
                                                        indent=2), encoding="utf-8")
    L = ["# run_all status", "", f"Generated {stamp} by `python -m experiments.run_all`. Complete steps are reused, "
         "never re-run. Numbers live in the linked summaries (real runs only).", ""]
    for e in EXPERIMENTS:
        L += [f"## Experiment {e.number}: {e.title}", ""]
        if e.summaries:
            L.append("Summaries: " + ", ".join(f"`results/{s}`" + ("" if (RES / s).exists() else " (missing)")
                                              for s in e.summaries))
            L.append("")
        L += ["| step | state | detail | command |", "|---|---|---|---|"]
        L += [f"| {r['step']} | {r['state']} | {r['detail']} | `{r['command']}` |"
              for r in rows if r["experiment"] == e.number]
        L.append("")
    if log:
        L += ["## Steps started by this invocation", "", "| step | exit code | seconds |", "|---|---|---|"]
        L += [f"| {x['step']} | {x['returncode']} | {x['seconds']} |" for x in log]
    (RES / "run_all_status.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-missing", action="store_true", help="run incomplete (non-manual) steps, one at a time")
    ap.add_argument("--min-free-mb", type=float, default=500.0)
    args = ap.parse_args()
    log = []
    if args.run_missing:
        import time

        for e in EXPERIMENTS:
            for s in e.steps:
                done, _ = s.status()
                if done or s.manual:
                    continue
                if free_ram_mb() < args.min_free_mb:
                    print(f"stop: free RAM {free_ram_mb():.0f} MB < {args.min_free_mb:.0f} MB before '{s.name}'")
                    write_status(collect(), log)
                    return
                t0 = time.perf_counter()
                print(f"run: {s.name}: {' '.join(s.command()[1:])}")
                rc = subprocess.run(s.command(), cwd=ROOT).returncode
                log.append({"step": s.name, "returncode": rc, "seconds": round(time.perf_counter() - t0, 1)})
                if rc != 0:
                    print(f"stop: '{s.name}' exited with {rc}")
                    write_status(collect(), log)
                    return
    rows = collect()
    write_status(rows, log)
    for r in rows:
        print(f"[{r['state']:<10}] {r['experiment']:<3} {r['step']:<38} {r['detail']}")


if __name__ == "__main__":
    main()
