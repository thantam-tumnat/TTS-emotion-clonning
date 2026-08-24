"""Run the voice-consistency test suites on demand and report them as a checklist.

The /tests page is a button that runs pytest and shows each test as a ✓/✗ line.
This is what backs it: a small, *fixed* set of suites (no arbitrary paths from the
client), each run in its own subprocess with a JUnit XML report that is parsed back
into structured results.

Only fast, mock-backed suites are listed -- none load the 8GB model, so a run is
seconds, not minutes. The GPU suite runs against the stub engine (SIANGTTS_GPU_STUB)
in its own virtualenv, so it is only offered when that project is present.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

# voice-cloning-with-tones/  (app/services/test_runner.py -> parents[2])
TONE_STUDIO_ROOT = Path(__file__).resolve().parents[2]
# Sibling project that hosts the GPU service and its stub-backed tests.
GPU_ROOT = TONE_STUDIO_ROOT.parent / "voice-cloning"


def _venv_python(root: Path) -> Optional[Path]:
    """The project's own interpreter, so each suite runs against its own deps."""
    for rel in ("Scripts/python.exe", "bin/python"):
        cand = root / ".venv" / rel
        if cand.exists():
            return cand
    return None


def _suite_defs() -> List[Dict[str, Any]]:
    """The fixed catalogue of runnable suites. Order is display order."""
    defs: List[Dict[str, Any]] = [
        {
            "id": "voice_consistency",
            "label": "เสียงเดิมทุก take (Tone Studio)",
            "description": "แต่ละ take ใช้เสียงคนเดิม — pinned speaker, auto-seed, และ anchor guard",
            "root": TONE_STUDIO_ROOT,
            "python": _venv_python(TONE_STUDIO_ROOT) or Path(sys.executable),
            "files": [
                "tests/test_take_voice_consistency.py",
                "tests/test_remote_take_voice_consistency.py",
            ],
            "env": {},
        },
        {
            "id": "benchmark",
            "label": "หน้าเทส / Benchmark API",
            "description": "session lifecycle, run-take, และ voice_anchor ที่ส่งกลับ",
            "root": TONE_STUDIO_ROOT,
            "python": _venv_python(TONE_STUDIO_ROOT) or Path(sys.executable),
            "files": ["tests/test_benchmark.py", "tests/test_chunk_split.py"],
            "env": {},
        },
    ]
    gpu_py = _venv_python(GPU_ROOT)
    if gpu_py and (GPU_ROOT / "tests" / "test_gpu_service.py").exists():
        defs.append(
            {
                "id": "gpu_service",
                "label": "GPU Service เคารพ logic (stub)",
                "description": "handle/seed ให้เสียงเดิมข้าม job — รันบน stub ไม่โหลดโมเดล",
                "root": GPU_ROOT,
                "python": gpu_py,
                "files": ["tests/test_gpu_service.py"],
                "env": {"SIANGTTS_GPU_STUB": "1", "SIANGTTS_STUB_DELAY": "0"},
            }
        )
    return defs


def list_suites() -> List[Dict[str, str]]:
    """What the page renders as selectable groups (no filesystem detail leaked)."""
    return [
        {"id": d["id"], "label": d["label"], "description": d["description"]}
        for d in _suite_defs()
    ]


def _parse_junit(xml_path: Path) -> List[Dict[str, Any]]:
    """Turn a JUnit report into one record per test.

    status is "passed" / "failed" / "error" / "skipped"; message carries the
    assertion text for the failures so the page can expand it.
    """
    tests: List[Dict[str, Any]] = []
    if not xml_path.exists():
        return tests
    root = ET.parse(xml_path).getroot()
    # <testsuites><testsuite><testcase/>...  -- findall reaches every testcase.
    for case in root.iter("testcase"):
        name = case.get("name", "?")
        status = "passed"
        message = ""
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is not None:
                status = "failed" if tag == "failure" else "error"
                message = (node.get("message") or "").strip() or (node.text or "").strip()
                break
        else:
            if case.find("skipped") is not None:
                status = "skipped"
        tests.append(
            {
                "name": name,
                "status": status,
                "duration_s": round(float(case.get("time", 0.0) or 0.0), 3),
                "message": message[:2000],
            }
        )
    return tests


def run_suite(suite_id: str) -> Dict[str, Any]:
    """Run one suite and return its checklist + counts."""
    defs = {d["id"]: d for d in _suite_defs()}
    suite = defs.get(suite_id)
    if suite is None:
        return {"id": suite_id, "label": suite_id, "error": "unknown suite", "tests": []}

    started = time.time()
    with tempfile.TemporaryDirectory() as td:
        xml_path = Path(td) / "report.xml"
        cmd = [
            str(suite["python"]), "-m", "pytest", *suite["files"],
            "-q", "-p", "no:cacheprovider", "--junitxml", str(xml_path),
        ]
        env = {**os.environ, **suite.get("env", {})}
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(suite["root"]),
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            tests = _parse_junit(xml_path)
            run_error = "" if tests else (proc.stderr or proc.stdout or "")[-2000:]
        except subprocess.TimeoutExpired:
            tests, run_error = [], "การรัน test เกินเวลาที่กำหนด (300s)"
        except Exception as e:  # noqa: BLE001
            tests, run_error = [], f"{type(e).__name__}: {e}"

    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for t in tests:
        counts[t["status"]] = counts.get(t["status"], 0) + 1

    return {
        "id": suite["id"],
        "label": suite["label"],
        "description": suite["description"],
        "tests": tests,
        "counts": counts,
        "total": len(tests),
        "ok": bool(tests) and counts["failed"] == 0 and counts["error"] == 0,
        "elapsed_s": round(time.time() - started, 2),
        "run_error": run_error,
    }


def run_all() -> Dict[str, Any]:
    """Run every suite; used by the page's one-click 'run all'."""
    suites = [run_suite(d["id"]) for d in _suite_defs()]
    return {
        "suites": suites,
        "ok": all(s["ok"] for s in suites),
        "total": sum(s["total"] for s in suites),
        "passed": sum(s["counts"].get("passed", 0) for s in suites),
        "failed": sum(s["counts"].get("failed", 0) + s["counts"].get("error", 0) for s in suites),
    }
