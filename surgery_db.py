"""SQLite storage for the demo surgery.

Everything here is fake demo data. The database file is created on first use
at data/surgery.db (delete it to reset the demo). Set SURGERY_DB_PATH to use a
different file, which the tests do.

Booking rules that must never be broken are enforced here in code and in the
database itself, not left to the model: a slot can only be booked once, a
patient can only touch their own appointments, and a patient cannot be in two
places at the same time.
"""

import datetime as dt
import difflib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import surgery_clock as clock

# --- Appointment types and slots ---------------------------------------------

APPT_TYPES = {
    "gp": "GP appointment",
    "nurse": "nurse appointment",
    "phone_consultation": "GP phone consultation",
    "blood_test": "blood test",
}

# Mock clinic rota: the same slots every weekday.
BASE_SLOTS = {
    "gp": ["09:00", "10:30", "14:00", "15:30"],
    "nurse": ["09:30", "11:00", "13:30", "16:00"],
    "phone_consultation": ["08:30", "12:00", "17:00"],
    "blood_test": ["08:00", "08:20", "08:40", "09:00"],
}

# The model sometimes says the same thing in a different way.
_TYPE_ALIASES = {
    "gp_appointment": "gp", "doctor": "gp", "gp_face_to_face": "gp",
    "nurse_appointment": "nurse",
    "phone": "phone_consultation", "telephone": "phone_consultation",
    "phone_appointment": "phone_consultation", "telephone_consultation": "phone_consultation",
    "gp_phone": "phone_consultation", "gp_phone_consultation": "phone_consultation",
    "blood": "blood_test", "bloods": "blood_test", "blood_tests": "blood_test",
    "phlebotomy": "blood_test",
}

# Everyday paperwork and enquiries. Lara records these for the team; she never actions them.
ADMIN_TYPES = {
    "fit_note": "Fit note",
    "test_results": "Test results enquiry",
    "referral": "Referral status",
    "letter_or_form": "Letter or form",
    "change_of_details": "Change of details",
    "other": "Other request",
}
_ADMIN_ALIASES = {
    "sick_note": "fit_note", "fitnote": "fit_note", "med3": "fit_note",
    "results": "test_results", "test_result": "test_results", "blood_results": "test_results",
    "referrals": "referral", "referral_status": "referral",
    "letter": "letter_or_form", "letters": "letter_or_form", "form": "letter_or_form",
    "forms": "letter_or_form", "certificate": "letter_or_form", "medical_report": "letter_or_form",
    "details": "change_of_details", "update_details": "change_of_details",
    "change_address": "change_of_details", "change_phone": "change_of_details",
    "contact_details": "change_of_details",
}

BOOKING_WINDOW_DAYS = 21
MIN_NOTICE_MINUTES = 15
NAME_MATCH_THRESHOLD = 0.8  # tolerant of small speech-recognition slips

SEED_PATIENTS = [
    # (name, date of birth, phone, demo appointment or None)
    # A demo appointment is (nth upcoming weekday, time, type, reason).
    # Names and 07700 900xxx numbers are fictional (Ofcom reserves that range for drama).
    ("Priya Sharma", "1985-03-14", "07700 900101", (1, "10:30", "gp", "follow-up appointment")),
    ("James Carter", "1972-11-02", "07700 900102", None),
    ("Aisha Khan", "1990-07-21", "07700 900103", None),
    ("Tom Williams", "2001-01-09", "07700 900104", None),
    ("Margaret Evans", "1948-05-30", "07700 900105", (2, "11:00", "nurse", "blood pressure check")),
    ("Sid", "2001-05-09", "07700 900106", (3, "15:30", "gp", "annual check-up")),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    phone TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    appt_date TEXT NOT NULL,
    appt_time TEXT NOT NULL,
    appt_type TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'booked',
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_active_slot
    ON appointments(appt_date, appt_time, appt_type) WHERE status = 'booked';
CREATE TABLE IF NOT EXISTS prescription_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    medication TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'requested',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS admin_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    request_type TEXT NOT NULL,
    details TEXT NOT NULL,
    phone_number TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS waiting_list (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    appt_type TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'waiting',
    slot_note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS callbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    urgency TEXT NOT NULL DEFAULT 'routine',
    patient_id INTEGER REFERENCES patients(id),
    caller_name TEXT NOT NULL DEFAULT '',
    phone_number TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL,
    preferred_time TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS call_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    patient_id INTEGER REFERENCES patients(id),
    summary_json TEXT NOT NULL
);
"""


def db_path() -> Path:
    override = os.getenv("SURGERY_DB_PATH")
    return Path(override) if override else Path(__file__).parent / "data" / "surgery.db"


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:  # commits on success, rolls back on error
            yield conn
    finally:
        conn.close()


def _now_text(now: dt.datetime | None = None) -> str:
    return (now or clock.uk_now()).replace(microsecond=0).isoformat()


def init_db(now: dt.datetime | None = None) -> None:
    """Create tables and make sure the demo patients exist.

    Safe to run on every start: a demo patient that is already there is left
    alone, and one that is missing (for example a newly added one) is added with
    their demo appointment. Nothing existing is changed or duplicated.
    """
    with connect() as conn:
        conn.executescript(_SCHEMA)
        today = clock.uk_today(now)
        weekdays = [
            today + dt.timedelta(days=i) for i in range(1, 15)
            if (today + dt.timedelta(days=i)).weekday() < 5
        ]
        for name, dob, phone, appointment in SEED_PATIENTS:
            exists = conn.execute(
                "SELECT 1 FROM patients WHERE full_name = ? AND date_of_birth = ?", (name, dob)
            ).fetchone()
            if exists:
                continue
            patient_id = conn.execute(
                "INSERT INTO patients (full_name, date_of_birth, phone) VALUES (?, ?, ?)",
                (name, dob, phone),
            ).lastrowid
            if appointment:
                day_index, time_text, appt_type, reason = appointment
                try:
                    conn.execute(
                        "INSERT INTO appointments (patient_id, appt_date, appt_time, appt_type, "
                        "reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (patient_id, weekdays[day_index].isoformat(), time_text, appt_type,
                         reason, _now_text(now)),
                    )
                except sqlite3.IntegrityError:
                    pass  # that slot is already taken; the patient simply has no demo booking


# --- Parsing the model's arguments ---------------------------------------------


def normalise_type(text: str) -> str | None:
    key = re.sub(r"[\s\-]+", "_", (text or "").strip().lower())
    key = _TYPE_ALIASES.get(key, key)
    return key if key in APPT_TYPES else None


def normalise_admin_type(text: str) -> str:
    key = re.sub(r"[\s\-]+", "_", (text or "").strip().lower())
    key = _ADMIN_ALIASES.get(key, key)
    return key if key in ADMIN_TYPES else "other"


def parse_date(text: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat((text or "").strip())
    except ValueError:
        return None


def parse_time(text: str) -> str | None:
    """Accept '14:00', '2pm', '2:30 pm' or '0930'; return 'HH:MM'."""
    match = re.match(
        r"^\s*(\d{1,2})(?:[:.]?(\d{2}))?\s*(am|pm)?\s*$", (text or "").lower()
    )
    if not match:
        return None
    hour, minute, meridiem = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def booking_ref(appt_id: int) -> str:
    return f"BK-{appt_id:04d}"


def parse_booking_ref(text: str) -> int | None:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else None


# --- Patients -----------------------------------------------------------------


def _normalise_name(name: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", "", (name or "").lower()).split())


def names_match(first: str, second: str) -> bool:
    """Same name allowing for small speech-recognition slips (Ayesha / Aisha)."""
    score = difflib.SequenceMatcher(None, _normalise_name(first), _normalise_name(second)).ratio()
    return score >= NAME_MATCH_THRESHOLD


def is_same_person(patient: dict, full_name: str, date_of_birth: dt.date | None) -> bool:
    """Do these details describe this already-verified patient?

    A different name, or a date of birth that clearly differs, means someone else.
    A date we cannot read is not held against them, so a badly formatted repeat of
    the same patient's details is not mistaken for a different person.
    """
    if not names_match(full_name, patient["full_name"]):
        return False
    return date_of_birth is None or date_of_birth.isoformat() == patient["date_of_birth"]


def find_patient(full_name: str, date_of_birth: dt.date) -> dict | None:
    """Date of birth must match exactly; the name may differ slightly."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE date_of_birth = ?", (date_of_birth.isoformat(),)
        ).fetchall()
    best, best_score = None, 0.0
    for row in rows:
        score = difflib.SequenceMatcher(
            None, _normalise_name(full_name), _normalise_name(row["full_name"])
        ).ratio()
        if score > best_score:
            best, best_score = dict(row), score
    return best if best_score >= NAME_MATCH_THRESHOLD else None


# --- Slots and appointments ------------------------------------------------------


def _appt_dict(row) -> dict:
    return {
        "booking_ref": booking_ref(row["id"]),
        "type": row["appt_type"],
        "type_label": APPT_TYPES.get(row["appt_type"], row["appt_type"]),
        "date": row["appt_date"],
        "time": row["appt_time"],
        "spoken_when": clock.speak_when(row["appt_date"], row["appt_time"]),
        "reason": row["reason"],
    }


def free_slots(appt_type: str, day: dt.date, now: dt.datetime | None = None) -> list[str]:
    now = now or clock.uk_now()
    today = now.date()
    if day < today or day > today + dt.timedelta(days=BOOKING_WINDOW_DAYS) or day.weekday() >= 5:
        return []
    with connect() as conn:
        booked = {
            row["appt_time"]
            for row in conn.execute(
                "SELECT appt_time FROM appointments "
                "WHERE appt_date = ? AND appt_type = ? AND status = 'booked'",
                (day.isoformat(), appt_type),
            )
        }
    cutoff = (now + dt.timedelta(minutes=MIN_NOTICE_MINUTES)).strftime("%H:%M") if day == today else "00:00"
    return [t for t in BASE_SLOTS[appt_type] if t not in booked and t > cutoff]


def next_available(appt_type: str, from_day: dt.date, count: int = 3,
                   now: dt.datetime | None = None) -> list[dict]:
    now = now or clock.uk_now()
    start = max(from_day, now.date())
    found = []
    for offset in range(BOOKING_WINDOW_DAYS + 1):
        day = start + dt.timedelta(days=offset)
        times = free_slots(appt_type, day, now)
        if times:
            found.append({
                "date": day.isoformat(),
                "spoken_date": clock.speak_date(day),
                "times": [{"time": t, "say": clock.format_time(t)} for t in times[:3]],
            })
            if len(found) == count:
                break
    return found


def upcoming_appointments(patient_id: int, now: dt.datetime | None = None) -> list[dict]:
    now = now or clock.uk_now()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE patient_id = ? AND status = 'booked' "
            "ORDER BY appt_date, appt_time",
            (patient_id,),
        ).fetchall()
    cutoff = now.replace(second=0, microsecond=0).isoformat(timespec="minutes")
    return [
        _appt_dict(r) for r in rows if f"{r['appt_date']}T{r['appt_time']}" > cutoff
    ]


def _error(code: str, message: str, **extra) -> dict:
    return {"ok": False, "error": code, "message": message, **extra}


def _patient_clash(conn, patient_id: int, day: str, time_text: str, ignore_id: int = 0) -> bool:
    row = conn.execute(
        "SELECT 1 FROM appointments WHERE patient_id = ? AND appt_date = ? AND appt_time = ? "
        "AND status = 'booked' AND id != ?",
        (patient_id, day, time_text, ignore_id),
    ).fetchone()
    return row is not None


def book(patient_id: int, appt_type: str, day: dt.date, time_text: str, reason: str,
         now: dt.datetime | None = None) -> dict:
    now = now or clock.uk_now()
    if time_text not in BASE_SLOTS[appt_type]:
        return _error("no_such_time", "That is not one of the appointment times for this type.",
                      next_available=next_available(appt_type, day, 3, now))
    if time_text not in free_slots(appt_type, day, now):
        return _error("slot_unavailable", "That slot is not available.",
                      next_available=next_available(appt_type, day, 3, now))
    try:
        with connect() as conn:
            if _patient_clash(conn, patient_id, day.isoformat(), time_text):
                return _error("patient_clash",
                              "This patient already has an appointment at that date and time.")
            cursor = conn.execute(
                "INSERT INTO appointments (patient_id, appt_date, appt_time, appt_type, "
                "reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (patient_id, day.isoformat(), time_text, appt_type, reason[:200], _now_text(now)),
            )
            row = conn.execute("SELECT * FROM appointments WHERE id = ?",
                               (cursor.lastrowid,)).fetchone()
    except sqlite3.IntegrityError:  # someone else took the slot a moment ago
        return _error("slot_unavailable", "That slot has just been taken.",
                      next_available=next_available(appt_type, day, 3, now))
    return {"ok": True, "appointment": _appt_dict(row)}


def _own_future_appointment(conn, patient_id: int, appt_id: int, now: dt.datetime):
    """Return the row only if it belongs to this patient and is still to come."""
    row = conn.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
    # Same answer whether it doesn't exist or belongs to someone else.
    if row is None or row["patient_id"] != patient_id:
        return None, _error("not_found", "No booked appointment found with that reference.")
    if row["status"] != "booked":
        return None, _error("not_booked", "That appointment has already been cancelled.")
    if f"{row['appt_date']}T{row['appt_time']}" <= now.isoformat(timespec="minutes"):
        return None, _error("in_the_past", "That appointment time has already passed.")
    return row, None


def _flag_waiting_list(conn, appt_type: str, day_text: str, time_text: str, freed_by: int) -> None:
    """A slot has just been freed: tell the staff desk who was waiting for that kind of appointment."""
    note = f"A {APPT_TYPES[appt_type]} slot has opened: {clock.speak_when(day_text, time_text)}."
    conn.execute(
        "UPDATE waiting_list SET status = 'slot_open', slot_note = ? "
        "WHERE appt_type = ? AND status IN ('waiting', 'slot_open') AND patient_id != ?",
        (note, appt_type, freed_by),
    )


def cancel(patient_id: int, appt_id: int, now: dt.datetime | None = None) -> dict:
    now = now or clock.uk_now()
    with connect() as conn:
        row, problem = _own_future_appointment(conn, patient_id, appt_id, now)
        if problem:
            return problem
        conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appt_id,))
        _flag_waiting_list(conn, row["appt_type"], row["appt_date"], row["appt_time"], patient_id)
    return {"ok": True, "cancelled": _appt_dict(row)}


def reschedule(patient_id: int, appt_id: int, new_day: dt.date, new_time: str,
               now: dt.datetime | None = None) -> dict:
    now = now or clock.uk_now()
    with connect() as conn:
        row, problem = _own_future_appointment(conn, patient_id, appt_id, now)
    if problem:
        return problem
    appt_type = row["appt_type"]
    if new_time not in BASE_SLOTS[appt_type] or new_time not in free_slots(appt_type, new_day, now):
        return _error("slot_unavailable", "That new slot is not available.",
                      next_available=next_available(appt_type, new_day, 3, now))
    try:
        with connect() as conn:
            if _patient_clash(conn, patient_id, new_day.isoformat(), new_time, ignore_id=appt_id):
                return _error("patient_clash",
                              "This patient already has an appointment at that date and time.")
            conn.execute(
                "UPDATE appointments SET appt_date = ?, appt_time = ? WHERE id = ?",
                (new_day.isoformat(), new_time, appt_id),
            )
            # Moving an appointment frees the old slot for someone on the waiting list.
            _flag_waiting_list(conn, appt_type, row["appt_date"], row["appt_time"], patient_id)
            updated = conn.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
    except sqlite3.IntegrityError:
        return _error("slot_unavailable", "That new slot has just been taken.",
                      next_available=next_available(appt_type, new_day, 3, now))
    return {"ok": True, "previous": _appt_dict(row), "appointment": _appt_dict(updated)}


# --- Prescriptions, callbacks, summaries -------------------------------------------


def add_prescription_request(patient_id: int, medication: str) -> int:
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO prescription_requests (patient_id, medication, created_at) "
            "VALUES (?, ?, ?)",
            (patient_id, medication[:120], _now_text()),
        )
        return cursor.lastrowid


def add_callback(kind: str, urgency: str, reason: str, patient_id: int | None = None,
                 caller_name: str = "", phone_number: str = "",
                 preferred_time: str = "") -> int:
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO callbacks (created_at, kind, urgency, patient_id, caller_name, "
            "phone_number, reason, preferred_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now_text(), kind, urgency, patient_id, caller_name[:80], phone_number[:30],
             reason[:400], preferred_time[:80]),
        )
        return cursor.lastrowid


def attach_callback_details(callback_ids: list[int], patient_id: int | None = None,
                            caller_name: str = "", phone_number: str = "",
                            phone_if_missing: str = "", preferred_time: str = "",
                            extra_reason: str = "") -> None:
    """Add what we learned later in the call to requests already recorded.

    An escalation is often recorded before we know who is calling or how to reach
    them. Updating that one record, instead of adding a second one, means staff
    see a single request per person per concern.
    """
    with connect() as conn:
        for callback_id in callback_ids:
            row = conn.execute("SELECT * FROM callbacks WHERE id = ?", (callback_id,)).fetchone()
            if row is None:
                continue
            updates = {}
            if patient_id is not None and row["patient_id"] is None:
                updates["patient_id"] = patient_id
            if caller_name and not row["caller_name"]:
                updates["caller_name"] = caller_name[:80]
            if phone_number:  # a number the patient gave us wins over the one on file
                updates["phone_number"] = phone_number[:30]
            elif phone_if_missing and not row["phone_number"]:
                updates["phone_number"] = phone_if_missing[:30]
            if preferred_time:
                updates["preferred_time"] = preferred_time[:80]
            if extra_reason and extra_reason.strip().lower() not in row["reason"].lower():
                updates["reason"] = f"{row['reason']} Also: {extra_reason.strip()}"[:400]
            if updates:
                conn.execute(
                    "UPDATE callbacks SET " + ", ".join(f"{column} = ?" for column in updates)
                    + " WHERE id = ?",
                    (*updates.values(), callback_id),
                )


def add_admin_request(patient_id: int, request_type: str, details: str,
                      phone_number: str = "") -> int:
    with connect() as conn:
        return conn.execute(
            "INSERT INTO admin_requests (created_at, patient_id, request_type, details, phone_number) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now_text(), patient_id, request_type, details[:400], phone_number[:30]),
        ).lastrowid


def join_waiting_list(patient_id: int, appt_type: str, note: str = "") -> tuple[int, bool]:
    """Add the patient to the cancellation list. Returns (entry id, True if newly added).

    One entry per patient per appointment type: asking again just updates the note.
    """
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM waiting_list WHERE patient_id = ? AND appt_type = ? "
            "AND status IN ('waiting', 'slot_open')",
            (patient_id, appt_type),
        ).fetchone()
        if existing:
            if note:
                conn.execute("UPDATE waiting_list SET note = ? WHERE id = ?", (note[:200], existing["id"]))
            return existing["id"], False
        entry_id = conn.execute(
            "INSERT INTO waiting_list (created_at, patient_id, appt_type, note) VALUES (?, ?, ?, ?)",
            (_now_text(), patient_id, appt_type, note[:200]),
        ).lastrowid
        return entry_id, True


def save_call_summary(session_id: str, patient_id: int | None, summary: dict) -> int:
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO call_summaries (created_at, session_id, patient_id, summary_json) "
            "VALUES (?, ?, ?, ?)",
            (_now_text(), session_id, patient_id, json.dumps(summary)),
        )
        return cursor.lastrowid


# --- What the staff desk shows ---------------------------------------------------------

_URGENCY_RANK = {"emergency": 0, "urgent": 1, "routine": 2}


def staff_snapshot(now: dt.datetime | None = None) -> dict:
    now = now or clock.uk_now()
    with connect() as conn:
        callbacks = [dict(r) for r in conn.execute(
            "SELECT c.*, p.full_name AS patient_name FROM callbacks c "
            "LEFT JOIN patients p ON p.id = c.patient_id WHERE c.status = 'open'"
        )]
        callbacks.sort(key=lambda r: r["created_at"], reverse=True)
        callbacks.sort(key=lambda r: _URGENCY_RANK.get(r["urgency"], 3))
        appointments = [dict(r) for r in conn.execute(
            "SELECT a.*, p.full_name AS patient_name FROM appointments a "
            "JOIN patients p ON p.id = a.patient_id WHERE a.status = 'booked' "
            "ORDER BY a.appt_date, a.appt_time"
        )]
        cutoff = now.isoformat(timespec="minutes")
        appointments = [a for a in appointments if f"{a['appt_date']}T{a['appt_time']}" > cutoff]
        prescriptions = [dict(r) for r in conn.execute(
            "SELECT r.*, p.full_name AS patient_name FROM prescription_requests r "
            "JOIN patients p ON p.id = r.patient_id WHERE r.status = 'requested' "
            "ORDER BY r.created_at DESC"
        )]
        admin_requests = [dict(r) for r in conn.execute(
            "SELECT a.*, p.full_name AS patient_name FROM admin_requests a "
            "JOIN patients p ON p.id = a.patient_id WHERE a.status = 'open' "
            "ORDER BY a.created_at DESC"
        )]
        waiting_list = [dict(r) for r in conn.execute(
            "SELECT w.*, p.full_name AS patient_name FROM waiting_list w "
            "JOIN patients p ON p.id = w.patient_id WHERE w.status IN ('waiting', 'slot_open') "
            "ORDER BY CASE w.status WHEN 'slot_open' THEN 0 ELSE 1 END, w.created_at"
        )]
        summaries = []
        for r in conn.execute(
            "SELECT s.*, p.full_name AS patient_name FROM call_summaries s "
            "LEFT JOIN patients p ON p.id = s.patient_id ORDER BY s.id DESC LIMIT 8"
        ):
            item = dict(r)
            item["summary"] = json.loads(item.pop("summary_json"))
            summaries.append(item)
    return {
        "generated_at": _now_text(now),
        "callbacks": callbacks,
        "appointments": appointments,
        "prescriptions": prescriptions,
        "admin_requests": admin_requests,
        "waiting_list": waiting_list,
        "summaries": summaries,
    }