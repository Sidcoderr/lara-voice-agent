"""Put the demo back to a clean start.

    uv run python reset_demo.py

Clears every request, booking, call summary and waiting-list entry from the demo
database, then puts the demo patients and their sample appointments back. It
works while the bot is running, so the staff page is clean before you record.
It only ever touches the demo database (data/surgery.db), never anything else.
"""

import surgery_db as db

# Children before parents, so nothing refers to a row that is already gone.
_TABLES = ["call_summaries", "prescription_requests", "admin_requests", "waiting_list",
           "callbacks", "appointments", "patients"]


def reset_demo() -> dict:
    db.init_db()  # make sure every table exists first
    with db.connect() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in _TABLES:
            conn.execute(f"DELETE FROM {table}")
        try:
            conn.execute("DELETE FROM sqlite_sequence")  # so booking references start again at BK-0001
        except Exception:  # noqa: BLE001
            pass
    db.init_db()  # the demo patients and their sample appointments
    with db.connect() as conn:
        return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _TABLES}


if __name__ == "__main__":
    counts = reset_demo()
    print(f"Demo reset ({db.db_path()}).")
    print(f"  {counts['patients']} demo patients and {counts['appointments']} sample appointments restored.")
    print("  The staff page is clean: no requests, callbacks or call summaries.")
