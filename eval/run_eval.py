"""
Runs the curated evaluation set (eval/cases.py) against the real system and
reports, per case: accuracy (pass/fail), latency, and an approximate
token-cost estimate.

This is a lightweight, dependency-free evaluation approach rather than an
LLM-as-judge setup:
  - Accuracy is checked via expected-keyword presence, which is reliable
    here because the ground truth (the Ema fixture, the real policy PDF's
    actual content) is either fully deterministic or was directly verified
    against the document before being written into a test case - not
    because keyword matching is generally a robust way to grade LLM output.
  - Token counts are a rough character-based estimate (~4 chars/token),
    not exact billed usage - getting exact usage would mean threading a
    callback/usage object through sql_agent.py/rag_agent.py/graph.py's
    internal .invoke() calls, which felt like too invasive a change to
    already-tested code just for this reporting feature.

Run with (requires DEEPSEEK_API_KEY in .env):
    venv\\Scripts\\python.exe eval/run_eval.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

from cases import EVAL_CASES
from graph import ask as ask_graph
from rag_agent import answer_policy_question
from sql_agent import answer_sql_question

CHARS_PER_TOKEN_ESTIMATE = 4  # rough rule-of-thumb, not an exact tokenizer


def _estimate_tokens(*texts: str) -> int:
    """Very rough token-count estimate (chars / 4), used only as a relative
    cost proxy across cases - not an exact billed token count."""
    return sum(len(text) for text in texts) // CHARS_PER_TOKEN_ESTIMATE


def _run_sql_case(case: dict) -> str:
    """Run a case through the SQL agent directly."""
    return answer_sql_question(case["question"])["answer"]


def _run_rag_case(case: dict) -> str:
    """Run a case through the RAG agent directly."""
    return answer_policy_question(case["question"])["answer"]


def _run_graph_case(case: dict) -> str:
    """Run a case through the full ReAct agent, including any setup turns
    for multi-turn/memory test cases."""
    thread_id = case.get("thread_id", case["id"])
    for setup_question in case.get("setup_questions", []):
        ask_graph(setup_question, thread_id=thread_id)
    return ask_graph(case["question"], thread_id=thread_id)


CASE_RUNNERS = {"sql": _run_sql_case, "rag": _run_rag_case, "graph": _run_graph_case}


def _grade(case: dict, answer: str) -> bool:
    """Check the answer contains every expected keyword."""
    return all(
        keyword.lower() in answer.lower() for keyword in case.get("expected_keywords", [])
    )


def run_case(case: dict) -> dict:
    """Execute one eval case against the real system and grade it."""
    start = time.perf_counter()
    answer = CASE_RUNNERS[case["agent"]](case)
    latency_seconds = time.perf_counter() - start

    return {
        "id": case["id"],
        "passed": _grade(case, answer),
        "latency_seconds": round(latency_seconds, 2),
        "approx_tokens": _estimate_tokens(case["question"], answer),
        "answer": answer,
    }


def run_eval() -> list:
    """Run every case in EVAL_CASES and return their graded results."""
    return [run_case(case) for case in EVAL_CASES]


def print_report(results: list) -> None:
    """Print a compact pass/fail table plus an overall summary."""
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['id']:28s} "
            f"latency={result['latency_seconds']:.2f}s "
            f"~tokens={result['approx_tokens']}"
        )
        if not result["passed"]:
            print(f"         answer: {result['answer'][:200]}")

    passed_count = sum(r["passed"] for r in results)
    total = len(results)
    total_latency = sum(r["latency_seconds"] for r in results)
    total_tokens = sum(r["approx_tokens"] for r in results)
    print()
    print(
        f"Passed {passed_count}/{total} "
        f"| total latency {total_latency:.2f}s "
        f"| ~{total_tokens} tokens total (approximate)"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print_report(run_eval())
