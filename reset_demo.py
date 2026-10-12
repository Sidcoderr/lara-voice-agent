"""Put the demo back to a clean start.

    uv run python reset_demo.py
    uv run python reset_demo.py --me "Siddharth Dangeti"

Clears every request, booking, call summary and waiting-list entry from the demo
database, then puts the demo patients and their sample appointments back. It
works while the bot is running, so the staff page is clean before you record.
It only ever touches the demo database (data/surgery.db), never anything else.

--me renames the demo patient who is normally called "Sid" (date of birth 9 May
2001), so that you can play yourself in the demo with your full name. The date
of birth does not change. Run it again without --me to go back to "Sid". Say the
full name when you verify: the date of birth must match exactly, and the name
must be close to it ("Siddharth" on its own is not close enough).
"""

import argparse
import re

import surgery_db as db

# Children before parents, so nothing refers to a row that is already gone.
_TABLES = ["call_summaries", "prescription_requests", "admin_requests", "waiting_list",
           "callbacks", "appointments", "patients"]

DEMO_PATIENT = "Sid"
_VALID_NAME = re.compile(r"[A-Za-z][A-Za-z '\-]{0,58}")


def reset_demo(me: str | None = None) -> dict:
    name = " ".join((me or "").split())
    if me is not None and not _VALID_NAME.fullmatch(name):
        raise ValueError("The name should be letters, spaces, apostrophes or hyphens only (up to 59 characters).")
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
    if name:
        with db.connect() as conn:
            conn.execute("UPDATE patients SET full_name = ? WHERE full_name = ?", (name, DEMO_PATIENT))
    with db.connect() as conn:
        return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _TABLES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--me", help='your full name, to replace the demo patient "Sid" (date of birth 9 May 2001)')
    args = parser.parse_args()
    try:
        counts = reset_demo(args.me)
    except ValueError as error:
        raise SystemExit(str(error))
    print(f"Demo reset ({db.db_path()}).")
    print(f"  {counts['patients']} demo patients and {counts['appointments']} sample appointments restored.")
    print("  The staff page is clean: no requests, callbacks or call summaries.")
    if args.me:
        print(f'  The demo patient is now "{" ".join(args.me.split())}" (date of birth 9 May 2001). Say the full name when you verify.')