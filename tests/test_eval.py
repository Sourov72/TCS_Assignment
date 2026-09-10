"""
Runs the curated evaluation set (eval/cases.py, eval/run_eval.py) as an
automated regression check against the real DeepSeek API. Skipped
automatically when no DEEPSEEK_API_KEY is configured, so CI without a key
still passes - same pattern as the MCP and Streamlit live integration tests.

Run with (requires DEEPSEEK_API_KEY in .env):
    venv\\Scripts\\python.exe -m pytest tests/test_eval.py -v -s
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from config import DEEPSEEK_API_KEY  # noqa: E402
from run_eval import print_report, run_eval  # noqa: E402


@pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="requires a real DEEPSEEK_API_KEY")
def test_eval_suite_passes_completely():
    """Every case in the curated eval set should pass against the real
    system - a failure here means a real regression in answer quality,
    routing, or retrieval faithfulness, not a flaky/unrelated test issue."""
    results = run_eval()
    print_report(results)  # -s flag shows this output for debugging a failure

    failed = [r["id"] for r in results if not r["passed"]]
    assert failed == [], f"Eval cases failed: {failed}"
