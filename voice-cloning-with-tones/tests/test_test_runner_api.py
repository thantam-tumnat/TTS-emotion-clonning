"""The /tests checklist page and the API behind it.

The runner itself shells out to pytest; these tests mock that boundary so they
never spawn a nested pytest run -- they check the page serves, the suite
catalogue is exposed, JUnit XML is parsed into a checklist, and the run endpoint
returns what a suite produced.
"""
from fastapi.testclient import TestClient

from app.main import app
from app.services import test_runner

client = TestClient(app)


def test_tests_page_serves():
    res = client.get("/tests")
    assert res.status_code == 200
    assert "Test Checklist" in res.text


def test_suite_catalogue_is_exposed():
    res = client.get("/api/tests/suites")
    assert res.status_code == 200
    ids = {s["id"] for s in res.json()["suites"]}
    # The two tone-studio suites are always present; gpu_service is optional.
    assert {"voice_consistency", "benchmark"} <= ids
    for s in res.json()["suites"]:
        assert s["label"] and s["description"]


def test_junit_is_parsed_into_a_checklist(tmp_path):
    xml = tmp_path / "r.xml"
    xml.write_text(
        """<?xml version="1.0"?>
        <testsuites><testsuite tests="3">
          <testcase classname="tests.t" name="test_ok" time="0.12"/>
          <testcase classname="tests.t" name="test_bad" time="0.03">
            <failure message="assert 1 == 2">E   assert 1 == 2</failure>
          </testcase>
          <testcase classname="tests.t" name="test_skip" time="0.0">
            <skipped message="no gpu"/>
          </testcase>
        </testsuite></testsuites>""",
        encoding="utf-8",
    )
    tests = test_runner._parse_junit(xml)
    by_name = {t["name"]: t for t in tests}
    assert by_name["test_ok"]["status"] == "passed"
    assert by_name["test_bad"]["status"] == "failed"
    assert "assert 1 == 2" in by_name["test_bad"]["message"]
    assert by_name["test_skip"]["status"] == "skipped"


def test_run_endpoint_returns_a_suite_result(monkeypatch):
    fake = {
        "id": "voice_consistency",
        "label": "x",
        "description": "y",
        "tests": [{"name": "test_a", "status": "passed", "duration_s": 0.1, "message": ""}],
        "counts": {"passed": 1, "failed": 0, "error": 0, "skipped": 0},
        "total": 1,
        "ok": True,
        "elapsed_s": 0.1,
        "run_error": "",
    }
    monkeypatch.setattr(test_runner, "run_suite", lambda sid: fake)

    res = client.post("/api/tests/run?suite_id=voice_consistency")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["tests"][0]["name"] == "test_a"


def test_run_all_endpoint_aggregates(monkeypatch):
    monkeypatch.setattr(
        test_runner,
        "run_all",
        lambda: {"suites": [], "ok": True, "total": 5, "passed": 5, "failed": 0},
    )
    res = client.post("/api/tests/run")
    assert res.status_code == 200
    assert res.json()["passed"] == 5
