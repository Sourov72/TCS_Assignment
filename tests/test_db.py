"""
Tests for the database schema + seeding logic (db/schema.sql, db/seed_db.py).

Run with:
    venv\\Scripts\\python.exe -m pytest tests/test_db.py -v
"""

import sqlite3
import sys
from pathlib import Path

# Make db/ importable as a plain module when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "db"))

from seed_db import seed_database  # noqa: E402  (import after sys.path tweak, by design)


def _fresh_seeded_db(tmp_path: Path) -> Path:
    """Seed a brand-new SQLite file inside pytest's temp dir and return its path.

    Using a temp file (instead of the real db/support.db) keeps tests isolated
    and safe to re-run without touching the actual project database.
    """
    db_path = tmp_path / "test_support.db"
    seed_database(db_path)
    return db_path


def test_seeding_creates_expected_customer_count(tmp_path):
    """25 customers total: 1 hand-written 'Ema' fixture + 24 random ones."""
    db_path = _fresh_seeded_db(tmp_path)
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    conn.close()
    assert count == 25


def test_seeding_creates_tickets(tmp_path):
    """Every seeded customer should result in at least some tickets overall."""
    db_path = _fresh_seeded_db(tmp_path)
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
    conn.close()
    assert count > 0


def test_ema_fixture_exists_with_full_ticket_history(tmp_path):
    """Ema is the assignment's example customer - verify she's present with
    her complete, predictable ticket history (used later as an eval fixture)."""
    db_path = _fresh_seeded_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    ema = conn.execute(
        "SELECT * FROM customers WHERE full_name = 'Ema Thompson'"
    ).fetchone()
    assert ema is not None
    assert ema["plan_tier"] == "Pro"

    tickets = conn.execute(
        "SELECT * FROM support_tickets WHERE customer_id = ?", (ema["customer_id"],)
    ).fetchall()
    conn.close()

    assert len(tickets) == 4
    categories = {t["category"] for t in tickets}
    assert categories == {"Billing", "Technical", "Refund", "Shipping"}


def test_seeding_is_idempotent(tmp_path):
    """Running seed_database twice on the same file should not duplicate rows."""
    db_path = tmp_path / "test_support.db"
    seed_database(db_path)
    seed_database(db_path)  # re-run on purpose

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    conn.close()
    assert count == 25
