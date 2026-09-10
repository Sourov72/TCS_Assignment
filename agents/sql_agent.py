"""
Natural-language question -> SQL query -> natural-language answer, over the
customers/support_tickets SQLite database (see db/schema.sql).

Pipeline:
  1. generate_sql_query(question): LLM writes a SQLite SELECT query, given
     the schema.
  2. is_safe_select_query(sql): guardrail - only a single, non-modifying
     SELECT statement is ever allowed through.
  3. execute_sql_query(sql): run it over a READ-ONLY database connection
     (belt-and-suspenders on top of the guardrail above - even if a bad
     query slipped past step 2, the DB layer itself would still refuse it).
  4. summarize_results(question, rows): LLM turns the raw rows into a
     friendly, concise answer.

Run with (manual smoke test, requires DEEPSEEK_API_KEY in .env):
    venv\\Scripts\\python.exe agents/sql_agent.py "Give me an overview of customer Ema's profile and past support ticket details."
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).parent.parent / "db"))

from init_db import DEFAULT_DB_PATH
from llm import get_chat_model

# Kept as a plain description (mirrors db/schema.sql) rather than introspecting
# the DB at import time, so this module has no hard dependency on the database
# file existing just to be imported. If the schema changes, update both files.
SCHEMA_DESCRIPTION = """\
customers(
    customer_id INTEGER PRIMARY KEY,
    full_name TEXT,
    email TEXT,
    phone TEXT,
    signup_date DATE,
    plan_tier TEXT,        -- Free, Basic, Pro, Enterprise
    location TEXT
)

support_tickets(
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER,   -- foreign key -> customers.customer_id
    subject TEXT,
    description TEXT,
    category TEXT,         -- Billing, Technical, Refund, Shipping, Account
    status TEXT,           -- Open, In Progress, Resolved, Closed
    priority TEXT,         -- Low, Medium, High, Urgent
    created_at TIMESTAMP,
    resolved_at TIMESTAMP, -- NULL if unresolved
    resolution_notes TEXT  -- NULL if unresolved
)
"""

SQL_GENERATION_PROMPT = """You are a SQLite expert. Given the schema below, \
write a single valid SQLite SELECT query that answers the user's question.

Schema:
{schema}

Rules:
- Output ONLY the raw SQL query - no explanation, no markdown code fences.
- Only ever write a SELECT statement. Never modify data.
- Use the table/column names exactly as given above.
- If the question asks about a customer by name, match with LIKE '%name%' \
(case-insensitive) rather than requiring an exact match.

Question: "{question}"

SQL query:"""

SUMMARY_PROMPT = """You are a helpful customer support assistant. Answer the \
user's question in a clear, friendly, concise way using ONLY the data below. \
If the data is empty, say plainly that no matching records were found - do \
not make anything up.

Question: "{question}"

Data (JSON rows from the database):
{rows_json}

Answer:"""

# Basic defense-in-depth guardrail. Even if this keyword check were somehow
# bypassed, execute_sql_query() also opens the database strictly read-only,
# so a data-modifying statement would still fail at the DB layer.
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "REPLACE", "ATTACH", "PRAGMA",
]

MAX_ROWS_FOR_SUMMARY = 30  # cap what we feed back into the LLM for summarization


def _strip_sql_fences(text: str) -> str:
    """Remove ```sql ... ``` markdown fences, in case the LLM adds them anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[3:] if text.lower().startswith("sql") else text
    return text.strip().rstrip(";")


def generate_sql_query(question: str) -> str:
    """Ask the LLM to translate a natural-language question into SQL."""
    prompt = SQL_GENERATION_PROMPT.format(schema=SCHEMA_DESCRIPTION, question=question)
    response = get_chat_model().invoke(prompt)
    return _strip_sql_fences(response.content)


def is_safe_select_query(sql: str) -> bool:
    """Guardrail: only a single, non-modifying SELECT statement is allowed.

    Forbidden keywords are matched as whole words (\\bKEYWORD\\b), not plain
    substrings - a naive substring check would false-positive on legitimate
    column names like "created_at" (contains "CREATE") or "updated_at"
    (contains "UPDATE").
    """
    normalized = sql.strip().rstrip(";")
    if ";" in normalized:  # reject stacked statements
        return False
    upper = normalized.upper()
    if not upper.startswith("SELECT"):
        return False
    return not any(
        re.search(rf"\b{keyword}\b", upper) for keyword in FORBIDDEN_KEYWORDS
    )


def execute_sql_query(sql: str, db_path: Path = DEFAULT_DB_PATH) -> list:
    """Run a SELECT query over a READ-ONLY connection, return rows as dicts."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def summarize_results(question: str, rows: list) -> str:
    """Turn raw SQL result rows into a friendly natural-language answer."""
    truncated_rows = rows[:MAX_ROWS_FOR_SUMMARY]
    rows_json = json.dumps(truncated_rows, default=str, indent=2)
    prompt = SUMMARY_PROMPT.format(question=question, rows_json=rows_json)
    response = get_chat_model().invoke(prompt)
    return response.content.strip()


def answer_sql_question(question: str) -> dict:
    """End-to-end: NL question -> SQL -> guarded execution -> NL answer.

    Returns the generated SQL and raw rows alongside the answer, so the UI
    can optionally show "how we got this answer" for transparency (mirrors
    the source-citation feature on the RAG side).
    """
    sql_query = generate_sql_query(question)

    if not is_safe_select_query(sql_query):
        return {
            "question": question,
            "sql_query": sql_query,
            "rows": [],
            "answer": "I can only answer questions that look up information, "
                      "not ones that would change data - I wasn't able to "
                      "safely run that query.",
        }

    try:
        rows = execute_sql_query(sql_query)
    except sqlite3.Error as error:
        return {
            "question": question,
            "sql_query": sql_query,
            "rows": [],
            "answer": f"I couldn't look that up due to a database error: {error}",
        }

    answer = summarize_results(question, rows)
    return {"question": question, "sql_query": sql_query, "rows": rows, "answer": answer}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or (
        "Give me an overview of customer Ema's profile and past support ticket details."
    )
    result = answer_sql_question(question)
    print(f"Question: {result['question']}")
    print(f"Generated SQL: {result['sql_query']}")
    print(f"Rows returned: {len(result['rows'])}")
    print(f"Answer:\n{result['answer']}")
