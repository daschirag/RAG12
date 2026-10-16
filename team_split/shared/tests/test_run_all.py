"""I3: run_all reuses finished results and never re-runs a complete step (subprocess is mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments import run_all as R


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "done.md").write_text("x", encoding="utf-8")
    (tmp_path / "part.jsonl").write_text("\n".join(json.dumps({"i": i}) for i in range(3)), encoding="utf-8")
    exps = [R.Experiment("1", "test", ("done.md",), [
        R.Step("finished", "experiments.finished", outputs=("done.md",)),
        R.Step("missing", "experiments.missing", outputs=("todo.md",)),
        R.Step("partial", "experiments.partial", outputs=("part.jsonl",), check=R._count("part.jsonl", 5, "i")),
        R.Step("needs decision", "experiments.manual", outputs=("never.md",), manual="needs a decision"),
    ])]
    monkeypatch.setattr(R, "RES", tmp_path)
    monkeypatch.setattr(R, "EXPERIMENTS", exps)
    started = []
    monkeypatch.setattr(R.subprocess, "run", lambda cmd, cwd: started.append(cmd[2]) or SimpleNamespace(returncode=0))
    return tmp_path, started


def test_status_only_starts_nothing(sandbox, monkeypatch):
    path, started = sandbox
    monkeypatch.setattr("sys.argv", ["run_all"])
    R.main()
    assert started == []
    rows = {r["step"]: r for r in json.loads((path / "run_all_status.json").read_text(encoding="utf-8"))["steps"]}
    assert rows["finished"]["state"] == "complete" and rows["missing"]["state"] == "incomplete"
    assert rows["partial"]["state"] == "incomplete" and rows["partial"]["detail"] == "3/5 items"
    assert rows["needs decision"]["state"] == "manual" and rows["needs decision"]["detail"] == "needs a decision"
    assert "| finished | complete |" in (path / "run_all_status.md").read_text(encoding="utf-8")


def test_run_missing_never_reruns_complete_steps(sandbox, monkeypatch):
    _, started = sandbox
    monkeypatch.setattr("sys.argv", ["run_all", "--run-missing"])
    monkeypatch.setattr(R, "free_ram_mb", lambda: 8000.0)
    R.main()
    assert started == ["experiments.missing", "experiments.partial"]       # not finished, not manual


def test_low_ram_stops_before_starting(sandbox, monkeypatch):
    _, started = sandbox
    monkeypatch.setattr("sys.argv", ["run_all", "--run-missing"])
    monkeypatch.setattr(R, "free_ram_mb", lambda: 300.0)
    R.main()
    assert started == []


def test_real_registry_is_well_formed():
    names = [s.name for e in R.EXPERIMENTS for s in e.steps]
    assert len(names) == len(set(names))
    assert [e.number for e in R.EXPERIMENTS] == ["1", "2", "3", "4", "5", "6", "7", "A-F"]
    for e in R.EXPERIMENTS:
        for s in e.steps:
            assert (R.ROOT / (s.module.replace(".", "/") + ".py")).is_file() or s.manual, s.module


def test_cloud_portability_is_pending_without_settings(tmp_path):
    from experiments.cloud_portability import REQUIRED, check

    res = check({}, tmp_path / ".env")
    assert res["status"] == "PENDING" and not res["usable"] and not any(res["settings_present"].values())
    full = check({k: "x" for k in REQUIRED}, tmp_path / ".env")
    assert full["usable"] and full["status"] == "READY"
    assert "x" not in json.dumps(full["settings_present"])          # values are never recorded
