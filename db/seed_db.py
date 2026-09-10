"""
Seeds the SQLite database with synthetic customer + support ticket data.

SYNTHETIC DATA DISCLOSURE
-------------------------
Everything this script writes to the database is fake:
- Most customers/tickets are randomly generated using the Faker library
  (https://faker.readthedocs.io), with a fixed random seed so the same
  data is produced every time this script is run.
- One customer, "Ema Thompson", is hand-written below (not Faker-generated)
  so she has a stable, predictable profile + ticket history. She matches
  the assignment's example query ("Give me an overview of customer Ema's
  profile and past support ticket details.") and doubles as a known-answer
  fixture for the evaluation script we'll add later.

No real person's data is used anywhere in this project.
"""

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

from init_db import DEFAULT_DB_PATH, init_db

# How much random filler data to generate alongside the Ema fixture.
NUM_RANDOM_CUSTOMERS = 24
MIN_TICKETS_PER_CUSTOMER = 1
MAX_TICKETS_PER_CUSTOMER = 5
RANDOM_SEED = 42  # fixed so re-running this script always produces the same data

PLAN_TIERS = ["Free", "Basic", "Pro", "Enterprise"]
CATEGORIES = ["Billing", "Technical", "Refund", "Shipping", "Account"]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]

# Short subject templates per category so generated tickets read naturally
# instead of like generic placeholder text.
SUBJECT_TEMPLATES = {
    "Billing": ["Charged twice for subscription", "Question about invoice #{n}", "Unexpected charge on card"],
    "Technical": ["Unable to log into account", "App crashes on startup", "Feature not working as expected"],
    "Refund": ["Refund request for order #{n}", "Haven't received refund yet", "Refund amount looks incorrect"],
    "Shipping": ["Package marked delivered but not received", "Order #{n} delayed", "Wrong item shipped"],
    "Account": ["Need to update account email", "Unable to reset password", "Request to close account"],
}


def _random_customer(fake: Faker) -> dict:
    """Build one plausible, randomized customer profile."""
    signup_date = fake.date_between(start_date="-3y", end_date="-30d")
    return {
        "full_name": fake.name(),
        "email": fake.unique.email(),
        "phone": fake.phone_number(),
        "signup_date": signup_date,
        "plan_tier": random.choice(PLAN_TIERS),
        "location": f"{fake.city()}, {fake.country()}",
    }


def _random_ticket(fake: Faker, customer_id: int, signup_date: date) -> dict:
    """Build one plausible, randomized support ticket for a given customer."""
    category = random.choice(CATEGORIES)
    subject = random.choice(SUBJECT_TEMPLATES[category]).format(n=random.randint(1000, 99999))
    status = random.choice(STATUSES)
    created_at = fake.date_time_between(start_date=signup_date, end_date="now")

    resolved_at, resolution_notes = None, None
    if status in ("Resolved", "Closed"):
        resolved_at = created_at + timedelta(days=random.randint(1, 10))
        resolution_notes = fake.sentence(nb_words=12)

    return {
        "customer_id": customer_id,
        "subject": subject,
        "description": fake.paragraph(nb_sentences=3),
        "category": category,
        "status": status,
        "priority": random.choice(PRIORITIES),
        "created_at": created_at,
        "resolved_at": resolved_at,
        "resolution_notes": resolution_notes,
    }


def _ema_customer() -> dict:
    """Hand-written fixture customer used in the assignment's example query."""
    return {
        "full_name": "Ema Thompson",
        "email": "ema.thompson@example.com",
        "phone": "+1-206-555-0148",
        "signup_date": date(2022, 3, 15),
        "plan_tier": "Pro",
        "location": "Seattle, USA",
    }


def _ema_tickets(customer_id: int) -> list:
    """Hand-written ticket history for Ema, so answers about her stay stable
    and predictable (used later as known-answer fixtures for evaluation)."""
    return [
        {
            "customer_id": customer_id,
            "subject": "Charged twice for subscription renewal",
            "description": "Ema noticed two identical charges for her Pro plan renewal on the same day.",
            "category": "Billing",
            "status": "Resolved",
            "priority": "High",
            "created_at": datetime(2024, 1, 10, 9, 30),
            "resolved_at": datetime(2024, 1, 12, 14, 0),
            "resolution_notes": "Duplicate charge confirmed and refunded within 2 business days.",
        },
        {
            "customer_id": customer_id,
            "subject": "Unable to reset password",
            "description": "Password reset email was not arriving; Ema was blocked from logging in.",
            "category": "Technical",
            "status": "Closed",
            "priority": "Medium",
            "created_at": datetime(2024, 4, 2, 16, 45),
            "resolved_at": datetime(2024, 4, 3, 10, 15),
            "resolution_notes": "Reset link was going to spam folder; manually reset password for the customer.",
        },
        {
            "customer_id": customer_id,
            "subject": "Refund request for canceled order #10432",
            "description": "Ordered an add-on service by mistake and requested cancellation within the refund window.",
            "category": "Refund",
            "status": "Resolved",
            "priority": "Medium",
            "created_at": datetime(2024, 7, 18, 11, 0),
            "resolved_at": datetime(2024, 7, 20, 9, 0),
            "resolution_notes": "Refund approved per policy since request was within 14 days; processed to original payment method.",
        },
        {
            "customer_id": customer_id,
            "subject": "Package marked delivered but not received",
            "description": "Tracking shows the package as delivered, but Ema says it never arrived at her address.",
            "category": "Shipping",
            "status": "Open",
            "priority": "High",
            "created_at": datetime(2024, 11, 5, 13, 20),
            "resolved_at": None,
            "resolution_notes": None,
        },
    ]


def _insert_customer(conn: sqlite3.Connection, customer: dict) -> int:
    """Insert one customer row and return its generated customer_id."""
    cursor = conn.execute(
        """
        INSERT INTO customers (full_name, email, phone, signup_date, plan_tier, location)
        VALUES (:full_name, :email, :phone, :signup_date, :plan_tier, :location)
        """,
        customer,
    )
    return cursor.lastrowid


def _insert_ticket(conn: sqlite3.Connection, ticket: dict) -> None:
    """Insert one support ticket row."""
    conn.execute(
        """
        INSERT INTO support_tickets
            (customer_id, subject, description, category, status, priority,
             created_at, resolved_at, resolution_notes)
        VALUES
            (:customer_id, :subject, :description, :category, :status, :priority,
             :created_at, :resolved_at, :resolution_notes)
        """,
        ticket,
    )


def seed_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Reset and repopulate the database: the Ema fixture + N random customers/tickets."""
    Faker.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    fake = Faker()

    init_db(db_path)  # make sure tables exist first

    conn = sqlite3.connect(db_path)
    try:
        # Clear existing rows so re-running this script is idempotent.
        conn.execute("DELETE FROM support_tickets")
        conn.execute("DELETE FROM customers")

        # Hand-written fixture customer (see _ema_customer docstring).
        ema_id = _insert_customer(conn, _ema_customer())
        for ticket in _ema_tickets(ema_id):
            _insert_ticket(conn, ticket)

        # Randomized filler customers + their ticket history.
        for _ in range(NUM_RANDOM_CUSTOMERS):
            customer = _random_customer(fake)
            customer_id = _insert_customer(conn, customer)
            num_tickets = random.randint(MIN_TICKETS_PER_CUSTOMER, MAX_TICKETS_PER_CUSTOMER)
            for _ in range(num_tickets):
                _insert_ticket(conn, _random_ticket(fake, customer_id, customer["signup_date"]))

        conn.commit()
    finally:
        conn.close()

    print(f"Seeded database at {db_path} with {NUM_RANDOM_CUSTOMERS + 1} customers (all synthetic).")


if __name__ == "__main__":
    seed_database()
