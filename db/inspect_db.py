"""
Quick manual sanity-check script for the seeded database.

Not a test suite (see tests/test_db.py for that) - just a convenient way for
a human to eyeball row counts and Ema's fixture record after seeding.

Run with:
    venv\\Scripts\\python.exe db/inspect_db.py
"""

import sqlite3

from init_db import DEFAULT_DB_PATH


def print_summary(db_path=DEFAULT_DB_PATH) -> None:
    """Print customer/ticket counts plus Ema's profile and ticket history."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    customer_count = cur.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]
    ticket_count = cur.execute("SELECT COUNT(*) AS n FROM support_tickets").fetchone()["n"]
    print(f"Total customers: {customer_count}")
    print(f"Total tickets:   {ticket_count}\n")

    ema = cur.execute("SELECT * FROM customers WHERE full_name = 'Ema Thompson'").fetchone()
    if ema is None:
        print("No 'Ema Thompson' fixture found - did you run db/seed_db.py?")
    else:
        print("Ema's profile:", dict(ema))
        print("Ema's tickets:")
        tickets = cur.execute(
            "SELECT subject, category, status, priority FROM support_tickets WHERE customer_id = ?",
            (ema["customer_id"],),
        ).fetchall()
        for ticket in tickets:
            print(" -", dict(ticket))

    conn.close()


if __name__ == "__main__":
    print_summary()
