"""A page where a patient can see what Lara booked and recorded for them: "My requests".

    http://localhost:7860/my-requests

Until now only staff could see requests (the /staff page). Lara says "I've recorded your repeat
prescription" and then the patient has nothing to look at. This page closes that gap: the patient
types their full name and date of birth, the same two checks Lara uses, and sees their OWN
appointments, repeat prescriptions, admin requests, cancellation-list place and messages for the team.

Rules (it is health information, so it is strict even in a demo):
  - Nothing appears without a match on date of birth (exact) and name (close), the same rule as the call.
  - Only that patient's records are ever shown.
  - The details go in a POST body, never in the address bar, so they are not saved in history or logs.
  - After 5 wrong attempts from one address in 10 minutes it stops answering for a while.
  - The failure message never says which detail was wrong.
  - It says plainly that a real service needs a proper login (for example NHS login).

PATIENT_PAGE=off turns the page off. If this module ever fails to load, Lara and /staff are unaffected.
"""

import datetime as dt
import html
import time
from urllib.parse import parse_qs

import request_updates as ru
import surgery_clock as clock
import surgery_db as db
from staff_desk import _CSS  # the same look as the reception desk

MAX_FAILURES = 5
WINDOW_SECONDS = 600
MAX_BODY_BYTES = 4096

_failures: dict[str, list[float]] = {}


# ------------------------------------------------------------------ attempt limiting


def _recent_failures(client: str, now: float) -> list[float]:
    recent = [t for t in _failures.get(client, []) if now - t < WINDOW_SECONDS]
    _failures[client] = recent
    return recent


def _blocked(client: str, now: float) -> bool:
    return len(_recent_failures(client, now)) >= MAX_FAILURES


def _record_failure(client: str, now: float) -> None:
    _recent_failures(client, now).append(now)


def reset_limits() -> None:
    _failures.clear()


# ------------------------------------------------------------------ the data


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def parse_form(body: bytes) -> dict:
    parsed = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items()}


def patient_items(patient_id: int, today: dt.date) -> dict:
    """Everything on record for ONE patient. Every query is limited to that patient_id."""
    with db.connect() as conn:
        appointments = conn.execute(
            "SELECT * FROM appointments WHERE patient_id = ? AND status = 'booked' AND appt_date >= ? "
            "ORDER BY appt_date, appt_time", (patient_id, today.isoformat())).fetchall()
        prescriptions = conn.execute(
            "SELECT * FROM prescription_requests WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        admin = conn.execute(
            "SELECT * FROM admin_requests WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        waiting = conn.execute(
            "SELECT * FROM waiting_list WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        messages = conn.execute(
            "SELECT * FROM callbacks WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
        items = {"appointments": [dict(r) for r in appointments], "prescriptions": [dict(r) for r in prescriptions],
                 "admin": [dict(r) for r in admin], "waiting": [dict(r) for r in waiting], "messages": [dict(r) for r in messages]}
        # The newest staff update for each request (in progress, done, and a message for the patient), if any.
        for key, kind in (("prescriptions", "prescription"), ("admin", "admin"), ("waiting", "waiting"), ("messages", "callback")):
            updates = ru.latest(conn, kind, [r["id"] for r in items[key]])
            for row in items[key]:
                row["update"] = updates.get(row["id"])
    return items


def _when(created_at: str, today: dt.date) -> str:
    stamp = dt.datetime.fromisoformat(created_at)
    time_text = clock.format_time(stamp.time())
    if stamp.date() == today:
        return f"Today, {time_text}"
    if stamp.date() == today - dt.timedelta(days=1):
        return f"Yesterday, {time_text}"
    return f"{stamp.day} {stamp.strftime('%b')}, {time_text}"


_BASE_STATUS = {
    "prescription": {"requested": "Requested. The team will review it."},
    "admin": {"open": "Received. The team will be in touch."},
    "waiting": {"waiting": "On the list"},
    "callback": {"open": "With the team"},
}
_PROGRESS = {"in_progress": "The team is working on it.", "done": "Done."}


def _status_text(kind: str, row: dict, today: dt.date) -> str:
    """What the patient reads in the Status column: the newest staff update if there is one, else the starting status."""
    update = row.get("update")
    if update:
        text = _PROGRESS.get(update["status"], update["status"].replace("_", " ").capitalize())
        if update["note"]:
            text += f" Message from the team: {update['note']}"
        when = _when(update["created_at"], today)
        if when.startswith(("Today", "Yesterday")):
            when = when[0].lower() + when[1:]
        return f"{text} (Updated {when})"
    if row["status"] == "done":
        return "Done."
    if kind == "waiting" and row["status"] == "slot_open":
        return f"A slot has opened: {row['slot_note']}. The team will contact you."
    return _BASE_STATUS[kind].get(row["status"], row["status"].replace("_", " ").capitalize())


def _message_kind(row: dict) -> str:
    return {"callback": "Call-back requested", "escalation": "Passed to a clinician"}.get(row["kind"], "Message for the team")


# ------------------------------------------------------------------ the pages


def _table(headers: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_e(empty)}</p>'
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


_EXTRA_CSS = """
form.lookup{margin-top:18px;background:var(--surface);padding:18px 20px;border-left:5px solid var(--brand);max-width:560px}
label{display:block;font-weight:600;margin:10px 0 4px}
input[type=text],input[type=date]{width:100%;padding:10px 12px;font:inherit;color:var(--ink);background:var(--bg);
border:1px solid var(--line);border-radius:4px}
button{margin-top:16px;padding:10px 20px;font:600 1rem Inter,system-ui,sans-serif;color:#fff;background:var(--brand);
border:0;border-radius:4px;cursor:pointer}
.error{background:var(--brick-bg);color:var(--brick-ink);padding:10px 14px;margin-top:14px;max-width:560px;border-left:5px solid var(--brick)}
form.again{margin-top:6px}
"""


def _shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{_e(title)}, Riverside Surgery</title>
<style>{_CSS}{_EXTRA_CSS}</style></head>
<body>
{body}
<footer>Demo only. This page asks for a name and date of birth, which is the same check Lara makes on a call. A real service would need
a proper login, for example NHS login. If you think you have a medical emergency, call 999.</footer>
</body></html>"""


def render_form(error: str = "", name: str = "") -> str:
    message = f'<p class="error" role="alert">{_e(error)}</p>' if error else ""
    body = f"""<header><h1>My requests</h1>
<p class="sub">Riverside Surgery. See what Lara booked or recorded for you.</p></header>
<main>{message}
<form class="lookup" method="post" action="/my-requests" autocomplete="off">
<label for="full_name">Your full name</label>
<input type="text" id="full_name" name="full_name" value="{_e(name)}" maxlength="80" required>
<label for="dob">Your date of birth</label>
<input type="date" id="dob" name="dob" required>
<button type="submit">Show my requests</button>
</form>
<p class="muted" style="margin-top:14px">We never put your details in the address bar. We only show what is on record for you.</p></main>"""
    return _shell("My requests", body)


def render_results(patient: dict, items: dict, today: dt.date) -> str:
    first = patient["full_name"].split()[0] if patient["full_name"].split() else "there"
    appointments = _table(
        ["When", "Type", "Reason", "Reference", "Status"],
        [[clock.speak_when(a["appt_date"], a["appt_time"]), db.APPT_TYPES.get(a["appt_type"], a["appt_type"]), a["reason"],
          db.booking_ref(a["id"]), "Booked"] for a in items["appointments"]],
        "You have no upcoming appointments.")
    prescriptions = _table(
        ["Requested", "Medicine", "Status"],
        [[_when(p["created_at"], today), p["medication"], _status_text("prescription", p, today)] for p in items["prescriptions"]],
        "You have no repeat prescription requests.")
    admin = _table(
        ["Requested", "What", "Details", "Status"],
        [[_when(a["created_at"], today), db.ADMIN_TYPES.get(a["request_type"], a["request_type"]), a["details"],
          _status_text("admin", a, today)] for a in items["admin"]],
        "You have no other requests.")
    waiting_rows = [[_when(w["created_at"], today), db.APPT_TYPES.get(w["appt_type"], w["appt_type"]),
                     _status_text("waiting", w, today)] for w in items["waiting"]]
    waiting = _table(["Added", "Waiting for", "Status"], waiting_rows, "You are not on the cancellation list.")
    messages = _table(
        ["Sent", "What", "Status"],
        [[_when(m["created_at"], today), _message_kind(m), _status_text("callback", m, today)] for m in items["messages"]],
        "You have no messages waiting for the team.")
    body = f"""<header><h1>Hello, {_e(first)}</h1>
<p class="sub">Riverside Surgery. Here is what is on record for you.</p></header>
<main>
<section><h2>Appointments<span class="count">{len(items["appointments"])}</span></h2>{appointments}</section>
<section><h2>Repeat prescriptions<span class="count">{len(items["prescriptions"])}</span></h2>{prescriptions}</section>
<section><h2>Other requests<span class="count">{len(items["admin"])}</span></h2>{admin}</section>
<section><h2>Cancellation list<span class="count">{len(items["waiting"])}</span></h2>{waiting}</section>
<section><h2>Messages for the team<span class="count">{len(items["messages"])}</span></h2>{messages}</section>
<form class="again" method="post" action="/my-requests">
<input type="hidden" name="full_name" value="{_e(patient["full_name"])}">
<input type="hidden" name="dob" value="{_e(patient["date_of_birth"])}">
<button type="submit">Refresh</button>
</form></main>"""
    return _shell("My requests", body)


# ------------------------------------------------------------------ the lookup (pure, so it is easy to test)


def handle_lookup(body: bytes, client: str, now: float | None = None, today: dt.date | None = None) -> tuple[int, str]:
    """Returns (status code, page). Never raises for bad input."""
    now = time.monotonic() if now is None else now
    if len(body) > MAX_BODY_BYTES:
        return 413, render_form(error="That request was too large.")
    if _blocked(client, now):
        return 429, render_form(error="Too many attempts. Please wait a few minutes and try again.")
    fields = parse_form(body)
    name = " ".join(fields.get("full_name", "").split())[:80]
    try:
        dob = dt.date.fromisoformat(fields.get("dob", "").strip())
        valid_dob = dt.date(1900, 1, 1) <= dob <= clock.uk_today()
    except ValueError:
        dob, valid_dob = None, False
    patient = db.find_patient(name, dob) if name and valid_dob else None
    if patient is None:
        _record_failure(client, now)
        return 200, render_form(error="We couldn't match those details. Please check the spelling and the date of birth.", name=name)
    today = today or clock.uk_today()
    return 200, render_results(patient, patient_items(patient["id"], today), today)


def register_patient_routes(app) -> None:
    """Add /my-requests to the runner's web server."""
    from fastapi import Request
    from fastapi.responses import HTMLResponse

    headers = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex", "Referrer-Policy": "no-referrer"}

    @app.get("/my-requests", response_class=HTMLResponse)
    async def my_requests_form():
        return HTMLResponse(render_form(), headers=headers)

    @app.post("/my-requests", response_class=HTMLResponse)
    async def my_requests_lookup(request: Request):
        body = await request.body()
        client = request.client.host if request.client else "unknown"
        status, page = handle_lookup(body, client)
        return HTMLResponse(page, status_code=status, headers=headers)
