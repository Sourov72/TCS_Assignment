"""
Creates (or re-creates) the SQLite database file from db/schema.sql.

Kept as its own tiny module, separate from data seeding, so schema changes
and data changes can be made/re-run independently of each other.
"""

import sqlite3
from pathlib import Path

# Paths are relative to this file so the script works regardless of the
# current working directory it's run from.
DB_DIR = Path(__file__).parent
SCHEMA_PATH = DB_DIR / "schema.sql"
DEFAULT_DB_PATH = DB_DIR / "support.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Apply schema.sql to the given SQLite file, creating tables if needed."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(schema_sql)
        connection.commit()
    finally:
        connection.close()
    print(f"Database ready at: {db_path}")


if __name__ == "__main__":
    init_db()
