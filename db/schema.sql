-- Schema for the customer support SQLite database.
-- Two tables: customers (profile info) and support_tickets (past ticket history).
-- Kept deliberately simple/flat so the SQL agent can answer natural-language
-- questions with straightforward JOIN / WHERE / GROUP BY queries.

-- Customer profile information.
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    phone         TEXT,
    signup_date   DATE NOT NULL,
    plan_tier     TEXT NOT NULL,      -- e.g. Free, Basic, Pro, Enterprise
    location      TEXT,               -- e.g. "Austin, USA"
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Past support ticket history, one row per ticket, linked to a customer.
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id       INTEGER NOT NULL,
    subject           TEXT NOT NULL,
    description       TEXT,
    category          TEXT NOT NULL,   -- e.g. Billing, Technical, Refund, Shipping, Account
    status            TEXT NOT NULL,   -- Open, In Progress, Resolved, Closed
    priority          TEXT NOT NULL,   -- Low, Medium, High, Urgent
    created_at        TIMESTAMP NOT NULL,
    resolved_at       TIMESTAMP,       -- NULL if still unresolved
    resolution_notes  TEXT,            -- NULL if still unresolved
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Indexes to keep lookups by customer fast as the table grows.
CREATE INDEX IF NOT EXISTS idx_tickets_customer_id ON support_tickets (customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers (email);
