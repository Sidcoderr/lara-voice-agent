"""Staff updates to a request, so the patient can see progress on /my-requests.

Until now a request said "Received. The team will be in touch." forever, because nothing ever changed it.
A staff member can now mark a request IN PROGRESS or DONE and add a short message for the patient
("Your fit note is ready to collect from reception"). The patient sees the new status, the message and
when it changed, the next time they look at /my-requests.

  - in progress : recorded as an update only; the request stays on the staff page, because it is still open.
  - done        : also closes the request, so it leaves the staff page.

Updates live in their own small table (request_updates), so nothing existing changes shape. The history
is kept: the patient sees the latest update.
"""

import surgery_clock as clock
import surgery_db as db

# kind -> (table, which status staff still see as open)
KINDS = {
    "admin": ("admin_requests", "open"),
    "prescription": ("prescription_requests", "requested"),
    "callback": ("callbacks", "open"),
    "waiting": ("waiting_list", "waiting"),
}
STATUSES = ("in_progress", "done")
NOTE_LIMIT = 200

_SCHEMA = """CREATE TABLE IF NOT EXISTS request_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    request_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
)"""


def ensure_table(conn) -> None:
    conn.execute(_SCHEMA)


def add_update(kind: str, request_id, status: str, note: str = "", now=None) -> dict:
    """Record a staff update. Raises ValueError for bad input and LookupError if there is no such request."""
    if kind not in KINDS:
        raise ValueError(f"The kind must be one of: {', '.join(KINDS)}.")
    if status not in STATUSES:
        raise ValueError(f"The status must be one of: {', '.join(STATUSES)}.")
    try:
        request_id = int(request_id)
    except (TypeError, ValueError):
        raise ValueError("The request number must be a whole number.") from None
    table, _ = KINDS[kind]  # table names come only from this fixed list, never from user input
    note = " ".join((note or "").split())[:NOTE_LIMIT]
    with db.connect() as conn:
        ensure_table(conn)
        row = conn.execute(f"SELECT status FROM {table} WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            raise LookupError(f"There is no {kind} request number {request_id}.")
        if row["status"] == "done":
            raise ValueError(f"{kind} request {request_id} is already done.")
        stamp = (now or clock.uk_now()).replace(microsecond=0).isoformat()
        conn.execute("INSERT INTO request_updates (created_at, kind, request_id, status, note) VALUES (?, ?, ?, ?, ?)",
                     (stamp, kind, request_id, status, note))
        if status == "done":
            conn.execute(f"UPDATE {table} SET status = 'done' WHERE id = ?", (request_id,))
    return {"kind": kind, "request_id": request_id, "status": status, "note": note, "created_at": stamp}


def latest(conn, kind: str, ids: list[int]) -> dict:
    """The newest update for each of these requests, keyed by request id."""
    if not ids:
        return {}
    ensure_table(conn)
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT * FROM request_updates WHERE kind = ? AND request_id IN ({marks}) ORDER BY id", (kind, *ids)).fetchall()
    return {r["request_id"]: dict(r) for r in rows}  # later rows replace earlier ones


def open_items() -> list[dict]:
    """Everything staff still see as open, with the numbers needed to update it."""
    queries = {
        "admin": "SELECT a.id, p.full_name AS patient, a.request_type || ': ' || a.details AS what, a.created_at "
                 "FROM admin_requests a JOIN patients p ON p.id = a.patient_id WHERE a.status = 'open'",
        "prescription": "SELECT r.id, p.full_name AS patient, r.medication AS what, r.created_at "
                        "FROM prescription_requests r JOIN patients p ON p.id = r.patient_id WHERE r.status = 'requested'",
        "callback": "SELECT c.id, COALESCE(p.full_name, c.caller_name, 'Unknown') AS patient, c.kind || ': ' || c.reason AS what, c.created_at "
                    "FROM callbacks c LEFT JOIN patients p ON p.id = c.patient_id WHERE c.status = 'open'",
        "waiting": "SELECT w.id, p.full_name AS patient, 'waiting for ' || w.appt_type AS what, w.created_at "
                   "FROM waiting_list w JOIN patients p ON p.id = w.patient_id WHERE w.status IN ('waiting', 'slot_open')",
    }
    items = []
    with db.connect() as conn:
        for kind, sql in queries.items():
            items += [{"kind": kind, **dict(r)} for r in conn.execute(sql)]
    return sorted(items, key=lambda i: (i["created_at"], i["kind"], i["id"]))


def attach(snapshot: dict) -> dict:
    """Add each request's newest staff update (if any) under the key 'update', for the staff page.

    It never raises: if anything goes wrong the snapshot is returned unchanged, so the staff page looks exactly as it did before.
    """
    try:
        with db.connect() as conn:
            for key, kind in (("callbacks", "callback"), ("admin_requests", "admin"),
                              ("prescriptions", "prescription"), ("waiting_list", "waiting")):
                rows = snapshot.get(key) or []
                updates = latest(conn, kind, [row["id"] for row in rows])
                for row in rows:
                    row["update"] = updates.get(row["id"])
    except Exception:  # noqa: BLE001
        pass
    return snapshot