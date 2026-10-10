"""The reception desk: what Lara has handled, and what still needs a person.

Served by the same server as the call screen:
    http://localhost:7860/staff        the page (refreshes itself)
    http://localhost:7860/staff/data   the same information as JSON

Or print it in a terminal:  uv run python staff_desk.py

DEMO ONLY: there is no login. A real deployment would need authentication,
audit logging and proper access control, because this shows patient details.
"""

import datetime as dt
from html import escape

import surgery_clock as clock
import surgery_db as db

_URGENCY_LABEL = {"emergency": "Emergency", "urgent": "Urgent", "routine": "Routine"}


def _when(created_at: str, today: dt.date) -> str:
    stamp = dt.datetime.fromisoformat(created_at)
    time_text = clock.format_time(stamp.time())
    if stamp.date() == today:
        return f"Today, {time_text}"
    if stamp.date() == today - dt.timedelta(days=1):
        return f"Yesterday, {time_text}"
    return f"{stamp.day} {stamp.strftime('%b')}, {time_text}"


def _phone(text: str) -> str:
    """07700900777 -> 07700 900777 (UK mobiles); anything else is shown as given."""
    digits = "".join(ch for ch in (text or "") if ch.isdigit())
    if len(digits) == 11 and digits.startswith("0") and not (text or "").startswith("+"):
        return f"{digits[:5]}\u00a0{digits[5:]}"  # non-breaking space, so it never wraps
    return text or ""


def _e(value) -> str:
    return escape(str(value if value is not None else ""))


def _request_rows(callbacks: list[dict], today: dt.date) -> str:
    if not callbacks:
        return (
            '<p class="empty">Nothing is waiting. Escalations and callback requests from '
            "Lara's calls appear here.</p>"
        )
    rows = []
    for item in callbacks:
        who = item["patient_name"] or item["caller_name"] or "Caller not identified"
        kind = "Escalated to a clinician" if item["kind"] == "escalation" else "Asked for a call back"
        phone = _phone(item["phone_number"]) or "No number on record"
        extra = f' <span class="muted">Prefers: {_e(item["preferred_time"])}</span>' if item["preferred_time"] else ""
        rows.append(
            f'<li class="request {_e(item["urgency"])}">'
            f'<div class="when"><strong>{_e(_URGENCY_LABEL.get(item["urgency"], "Routine"))}</strong>'
            f'<span>{_e(_when(item["created_at"], today))}</span></div>'
            f'<div class="what"><p class="who">{_e(who)}. {_e(kind)}.</p>'
            f'<p>{_e(item["reason"])}</p>'
            f'<p class="muted">{_e(phone)}{extra}</p></div></li>'
        )
    return f'<ul class="requests">{"".join(rows)}</ul>'


def _waiting_rows(waiting: list[dict], today: dt.date) -> str:
    if not waiting:
        return '<p class="empty">Nobody is waiting for a cancellation.</p>'
    rows = []
    for item in waiting:
        free = item["status"] == "slot_open"
        label = "Slot free" if free else "Waiting"
        kind = db.APPT_TYPES.get(item["appt_type"], item["appt_type"])
        detail = _e(item["slot_note"]) if free else ""
        wants = f'<p class="muted">Prefers: {_e(item["note"])}</p>' if item["note"] else ""
        rows.append(
            f'<li class="request {"urgent" if free else "routine"}">'
            f'<div class="when"><strong>{label}</strong><span>{_e(_when(item["created_at"], today))}</span></div>'
            f'<div class="what"><p class="who">{_e(item["patient_name"])}. Waiting for a {_e(kind)}.</p>'
            f'{f"<p>{detail}</p>" if detail else ""}{wants}</div></li>'
        )
    return f'<ul class="requests">{"".join(rows)}</ul>'


def _table(headers: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_e(empty)}</p>'
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _call_rows(summaries: list[dict], today: dt.date) -> str:
    if not summaries:
        return '<p class="empty">No finished calls yet. A summary appears here when a call ends.</p>'
    items = []
    for item in summaries:
        s = item["summary"]
        who = item["patient_name"] or "Patient not verified"
        actions = "".join(f"<li>{_e(a)}</li>" for a in s.get("actions_taken", []))
        follow = (
            f'<p class="follow"><strong>Needs follow-up.</strong> {_e(s.get("follow_up_details") or "See open requests.")}</p>'
            if s.get("follow_up_needed") else ""
        )
        note = ' <span class="muted">(automatic summary unavailable)</span>' if s.get("source") == "fallback" else ""
        items.append(
            f'<li class="call"><div class="when"><strong>{_e(who)}</strong>'
            f'<span>{_e(_when(item["created_at"], today))}</span></div>'
            f'<div class="what"><p>{_e(s.get("reason_for_call"))}{note}</p>'
            f'<p class="muted">{_e(s.get("outcome"))}</p>'
            f'{"<ul>" + actions + "</ul>" if actions else ""}{follow}</div></li>'
        )
    return f'<ul class="calls">{"".join(items)}</ul>'


_CSS = """
:root{--bg:#EFF3EE;--surface:#FAFBF9;--ink:#1B2B29;--soft:#55655F;--line:#D5DDD7;--brand:#2F6F62;
--amber:#D9A441;--amber-ink:#7A5200;--amber-bg:#F6EBD2;--brick:#B5533C;--brick-ink:#8F3322;--brick-bg:#F3DDD6}
@media (prefers-color-scheme:dark){:root{--bg:#142220;--surface:#1B2C29;--ink:#E6EEEA;--soft:#9DB0A9;--line:#2C403B;
--brand:#6FB5A4;--amber-ink:#F0C46C;--amber-bg:#3A3018;--brick-ink:#F0A08C;--brick-bg:#3C201A}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 Inter,system-ui,-apple-system,Segoe UI,sans-serif}
header,main{max-width:960px;margin:0 auto;padding:0 20px}
header{padding-top:36px;padding-bottom:8px}
h1,h2{font-family:Fraunces,Georgia,serif;font-weight:500;margin:0}
h1{font-size:2rem}
h2{font-size:1.35rem;margin-bottom:6px}
.sub{margin:6px 0 0;color:var(--soft)}
.stamp{margin:4px 0 0;color:var(--soft);font-size:.85rem}
section{padding:28px 0 8px}
.count{font:600 .9rem Inter,system-ui,sans-serif;color:var(--soft);margin-left:6px}
.muted{color:var(--soft);font-size:.92rem}
.empty{margin:10px 0;color:var(--soft)}
p{margin:0 0 4px}
ul{list-style:none;margin:0;padding:0}
.request,.call{display:grid;grid-template-columns:150px 1fr;gap:16px;padding:14px 16px;margin-top:10px;
background:var(--surface);border-left:5px solid var(--line)}
.request.urgent{border-left-color:var(--amber);background:var(--amber-bg)}
.request.emergency{border-left-color:var(--brick);background:var(--brick-bg)}
.request.urgent strong{color:var(--amber-ink)}
.request.emergency strong{color:var(--brick-ink)}
.when{display:flex;flex-direction:column;font-size:.92rem}
.when span{color:var(--soft)}
.who{font-weight:600}
.call{border-left-color:var(--brand)}
.call ul{list-style:disc;padding-left:20px;margin:6px 0;font-size:.94rem}
.follow{margin-top:6px}
.scroll{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;background:var(--surface)}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--soft);font-size:.9rem}
footer{max-width:960px;margin:0 auto;padding:28px 20px 48px;color:var(--soft);font-size:.85rem}
@media (max-width:640px){.request,.call{grid-template-columns:1fr;gap:6px}}
"""


def render_page(snapshot: dict) -> str:
    now = dt.datetime.fromisoformat(snapshot["generated_at"])
    today = now.date()
    callbacks = snapshot["callbacks"]
    appointments = _table(
        ["When", "Patient", "Type", "Reason", "Reference"],
        [
            [clock.speak_when(a["appt_date"], a["appt_time"]), a["patient_name"],
             db.APPT_TYPES.get(a["appt_type"], a["appt_type"]), a["reason"], db.booking_ref(a["id"])]
            for a in snapshot["appointments"]
        ],
        "No upcoming appointments.",
    )
    admin = _table(
        ["Requested", "Patient", "Type", "Details", "Phone"],
        [[_when(a["created_at"], today), a["patient_name"], db.ADMIN_TYPES.get(a["request_type"], a["request_type"]),
          a["details"], _phone(a["phone_number"])] for a in snapshot["admin_requests"]],
        "No admin requests waiting.",
    )
    prescriptions = _table(
        ["Requested", "Patient", "Medicine"],
        [[_when(p["created_at"], today), p["patient_name"], p["medication"]]
         for p in snapshot["prescriptions"]],
        "No repeat prescription requests waiting.",
    )
    return f"""<!doctype html>
<html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="10">
<title>Reception desk, Riverside Surgery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>{_CSS}</style></head>
<body>
<header><h1>Reception desk</h1>
<p class="sub">Riverside Surgery. What Lara has handled, and what still needs a person.</p>
<p class="stamp">Updated {_e(clock.format_time(now.time()))}. This page refreshes itself every 10 seconds.</p></header>
<main>
<section><h2>Needs a person<span class="count">{len(callbacks)}</span></h2>{_request_rows(callbacks, today)}</section>
<section><h2>Admin requests<span class="count">{len(snapshot["admin_requests"])}</span></h2>{admin}</section>
<section><h2>Cancellation list<span class="count">{len(snapshot["waiting_list"])}</span></h2>{_waiting_rows(snapshot["waiting_list"], today)}</section>
<section><h2>Upcoming appointments<span class="count">{len(snapshot["appointments"])}</span></h2>{appointments}</section>
<section><h2>Repeat prescription requests<span class="count">{len(snapshot["prescriptions"])}</span></h2>{prescriptions}</section>
<section><h2>Recent calls</h2>{_call_rows(snapshot["summaries"], today)}</section>
</main>
<footer>Demo data only. This page has no login, so it must not be used with real patient information.</footer>
</body></html>"""


def register_staff_routes(app) -> None:
    """Add /staff and /staff/data to the runner's web server."""
    from fastapi.responses import HTMLResponse, JSONResponse

    db.init_db()

    @app.get("/staff", response_class=HTMLResponse)
    async def staff_page():
        return HTMLResponse(render_page(db.staff_snapshot()))

    @app.get("/staff/data")
    async def staff_data():
        return JSONResponse(db.staff_snapshot())


def print_summary() -> None:
    db.init_db()
    snap = db.staff_snapshot()
    print(f"\nNeeds a person ({len(snap['callbacks'])})")
    for c in snap["callbacks"] or [None]:
        if c:
            print(f"  [{c['urgency']}] {c['patient_name'] or c['caller_name'] or 'Unknown'}: {c['reason']} ({c['phone_number'] or 'no number'})")
        else:
            print("  nothing waiting")
    print(f"\nAdmin requests ({len(snap['admin_requests'])})")
    for a in snap["admin_requests"]:
        print(f"  {a['patient_name']}: {db.ADMIN_TYPES.get(a['request_type'])}: {a['details']}")
    print(f"\nCancellation list ({len(snap['waiting_list'])})")
    for w in snap["waiting_list"]:
        print(f"  {w['patient_name']}: {db.APPT_TYPES.get(w['appt_type'])} ({'SLOT FREE: ' + w['slot_note'] if w['status'] == 'slot_open' else 'waiting'})")
    print(f"\nUpcoming appointments ({len(snap['appointments'])})")
    for a in snap["appointments"]:
        print(f"  {clock.speak_when(a['appt_date'], a['appt_time'])}: {a['patient_name']}, {db.APPT_TYPES.get(a['appt_type'])}")
    print(f"\nRepeat prescription requests ({len(snap['prescriptions'])})")
    for p in snap["prescriptions"]:
        print(f"  {p['patient_name']}: {p['medication']}")
    print(f"\nRecent calls ({len(snap['summaries'])})")
    for s in snap["summaries"]:
        print(f"  {s['patient_name'] or 'Not verified'}: {s['summary'].get('reason_for_call')}")


if __name__ == "__main__":
    print_summary()