"""Offline tests for Lara's tools. No API keys or network needed.

Run from the project folder:
    uv run python tests/test_tools.py

Uses a throwaway database, so your real data/surgery.db is never touched.
"""

import asyncio
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

os.environ["SURGERY_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grounding  # noqa: E402
import surgery_clock as clock  # noqa: E402
import surgery_db as db  # noqa: E402
from tools import MAX_VERIFY_ATTEMPTS, SessionState, build_tools  # noqa: E402

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  -> {detail}" if detail and not condition else ""))


class FakeContext:
    """Stands in for the conversation: just what the patient has said so far."""

    def __init__(self, said):
        self.said = said

    def get_messages(self):
        return [{"role": "user", "content": line} for line in self.said]


class FakeParams:
    """Stands in for Pipecat's FunctionCallParams: just captures the tool's reply."""

    def __init__(self, said=None):
        self.reply = None
        # No context means "cannot read the conversation", so grounding is skipped.
        self.context = FakeContext(said) if said is not None else None

    async def result_callback(self, payload, *args, **kwargs):
        self.reply = payload


class Call:
    """One simulated phone call: its own state and tools."""

    def __init__(self):
        self.state = SessionState()
        self.tools = {fn.__name__: fn for fn in build_tools(self.state)}

    def use(self, tool: str, said=None, **kwargs):
        params = FakeParams(said)
        asyncio.run(self.tools[tool](params, **kwargs))
        return params.reply


def weekday(offset_weekdays: int) -> dt.date:
    """The Nth weekday after today (UK date)."""
    day, found = clock.uk_today(), 0
    while found < offset_weekdays:
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            found += 1
    return day


def next_saturday() -> dt.date:
    day = clock.uk_today()
    while day.weekday() != 5:
        day += dt.timedelta(days=1)
    return day


def test_clock():
    print("\nUK time and opening hours")
    utc = lambda *a: dt.datetime(*a, tzinfo=dt.timezone.utc)
    check("summer time is UTC+1", clock.uk_now(utc(2026, 7, 1, 12, 0)).hour == 13)
    check("winter time is UTC+0", clock.uk_now(utc(2026, 12, 1, 12, 0)).hour == 12)
    check("clocks go back at 01:00 UTC on 25 Oct 2026",
          clock.uk_now(utc(2026, 10, 25, 0, 59)).hour == 1 and clock.uk_now(utc(2026, 10, 25, 1, 0)).hour == 1)
    check("clocks go forward on 29 Mar 2026", clock.uk_now(utc(2026, 3, 29, 1, 0)).hour == 2)
    monday_10am, saturday_10am = dt.datetime(2026, 10, 5, 10, 0), dt.datetime(2026, 10, 10, 10, 0)
    check("open weekday morning", clock.is_open(monday_10am))
    check("closed before 8am", not clock.is_open(dt.datetime(2026, 10, 5, 7, 59)))
    check("closed from 6:30pm", not clock.is_open(dt.datetime(2026, 10, 5, 18, 30)))
    check("closed at weekends", not clock.is_open(saturday_10am))
    check("next opening from Saturday is Monday", "Monday" in clock.next_opening(saturday_10am))
    check("friendly times", clock.format_time("14:30") == "2:30pm" and clock.format_time("08:00") == "8am"
          and clock.format_time("12:00") == "12pm")
    check("friendly dates", clock.speak_date("2026-10-06") == "Tuesday 6 October")


def test_parsing():
    print("\nParsing what the model sends")
    check("time 2pm", db.parse_time("2pm") == "14:00")
    check("time 2:30 pm", db.parse_time("2:30 pm") == "14:30")
    check("time 14:00", db.parse_time("14:00") == "14:00")
    check("time nonsense", db.parse_time("teatime") is None)
    check("type aliases", db.normalise_type("Blood tests") == "blood_test" and db.normalise_type("doctor") == "gp"
          and db.normalise_type("phone consultation") == "phone_consultation")
    check("unknown type", db.normalise_type("dentist") is None)


def test_verification():
    print("\nVerifying the patient")
    call = Call()
    check("booking is refused before verification",
          call.use("book_appointment", date=weekday(2).isoformat(), time="14:00")["error"] == "not_verified")
    check("prescription is refused before verification",
          call.use("request_repeat_prescription", medication_name="X")["error"] == "not_verified")
    check("cancel is refused before verification",
          call.use("cancel_appointment", booking_ref="BK-0001")["error"] == "not_verified")
    bad = call.use("verify_patient", full_name="Priya Sharma", date_of_birth="1985-03-15")
    check("wrong date of birth fails", bad["verified"] is False and bad["attempts_left"] == MAX_VERIFY_ATTEMPTS - 1)
    check("failure does not reveal which detail was wrong", "name" not in bad["message"].lower().split("do not say")[0])
    check("bad date format is not counted as an attempt",
          call.use("verify_patient", full_name="Priya Sharma", date_of_birth="March")["error"] == "bad_date_format"
          and call.state.failed_verifications == 1)
    good = call.use("verify_patient", full_name="priya sharma", date_of_birth="1985-03-14")
    check("correct details verify (case-insensitive)", good["verified"] and good["first_name"] == "Priya")
    check("upcoming appointments are listed on verification", len(good["upcoming_appointments"]) == 1)
    check("small speech slip in the name still verifies",
          Call().use("verify_patient", full_name="Priya Sharmer", date_of_birth="1985-03-14")["verified"])
    check("a different name with a real DOB does not verify",
          not Call().use("verify_patient", full_name="Bob Smith", date_of_birth="1985-03-14")["verified"])
    locked = Call()
    for _ in range(MAX_VERIFY_ATTEMPTS):
        locked.use("verify_patient", full_name="Nobody Here", date_of_birth="1980-01-01")
    after = locked.use("verify_patient", full_name="Priya Sharma", date_of_birth="1985-03-14")
    check("locks out after repeated failures, even with right details", after.get("locked") is True)


def test_grounding():
    print("\nA date of birth must really have been said by the patient")
    d = dt.date
    said = grounding.dob_was_said
    check("spoken in words", said(d(1990, 7, 21), "my date of birth is the twenty first of July nineteen ninety"))
    check("ordinal with digits", said(d(1990, 7, 21), "21st of July 1990"))
    check("month first", said(d(1990, 7, 21), "July 21, 1990"))
    check("with slashes", said(d(1990, 7, 21), "it's 21/07/1990"))
    check("two-digit year", said(d(1990, 7, 21), "21-7-90"))
    check("year in words with 'and'", said(d(2001, 1, 9), "the ninth of January two thousand and one"))
    check("year as 'twenty oh one'", said(d(2001, 1, 9), "9th January twenty oh one"))
    check("the first of the month", said(d(1985, 3, 1), "the first of March nineteen eighty five"))
    check("said across several turns", said(d(1990, 7, 21), "my birthday is the twenty first. July. nineteen ninety"))
    check("only a year is not enough (the real failure in testing)", not said(d(2000, 1, 1), "And I'm born in two thousand."))
    check("name and year but no day or month", not said(d(1990, 7, 21), "I'm Ayesha Khan and I was born in 1990"))
    check("no date at all", not said(d(1990, 7, 21), "Hi Lara, my name is Sudhat"))
    check("wrong day", not said(d(1990, 7, 21), "22 July 1990"))
    check("wrong year", not said(d(1990, 7, 21), "21 July 1991"))
    check("empty", not said(d(1990, 7, 21), ""))

    print("\nverify_patient refuses a date of birth the patient never said")
    call = Call()
    invented = call.use("verify_patient", said=["Hi Lara, my name is Aisha Khan", "I'd like a GP appointment"],
                        full_name="Aisha Khan", date_of_birth="1990-07-21")
    check("a correct but unspoken date of birth is refused", invented["verified"] is False and invented["error"] == "date_of_birth_not_heard")
    check("a made-up attempt does not count against the patient", call.state.failed_verifications == 0)
    for _ in range(5):
        call.use("verify_patient", said=["I'm Aisha Khan"], full_name="Aisha Khan", date_of_birth="1990-07-21")
    check("repeated guessing never locks the patient out", call.state.failed_verifications == 0)
    partial = call.use("verify_patient", said=["I'm Aisha Khan", "born in nineteen ninety"], full_name="Aisha Khan", date_of_birth="1990-07-21")
    check("a year alone is not accepted", partial["verified"] is False)
    good = call.use("verify_patient", said=["I'm Aisha Khan", "born on the twenty first of July nineteen ninety"],
                    full_name="Aisha Khan", date_of_birth="1990-07-21")
    check("verifies once the patient has said it", good["verified"] is True)
    other = Call()
    guess = other.use("verify_patient", said=["I'm Priya Sharma", "I was born in March nineteen eighty five"],
                      full_name="Priya Sharma", date_of_birth="1985-03-14")
    check("a guessed day on a real record is refused", guess["verified"] is False and other.state.patient is None)
    wrong = Call().use("verify_patient", said=["Bob Smith", "14 March 1985"], full_name="Bob Smith", date_of_birth="1985-03-14")
    check("a said date with the wrong name still fails", wrong["verified"] is False and wrong.get("error") != "date_of_birth_not_heard")
    mishear = Call().use("verify_patient", said=["My name is Ayesha Khan", "21 July 1990"], full_name="Ayesha Khan", date_of_birth="1990-07-21")
    check("Ayesha matches Aisha when the date is right", mishear["verified"] is True)


def test_demo_patients():
    print("\nDemo patients")
    sid = Call().use("verify_patient", said=["My name is Sid", "I was born on the ninth of May two thousand and one"],
                     full_name="Sid", date_of_birth="2001-05-09")
    check("Sid verifies with the right date of birth", sid["verified"] is True and sid["first_name"] == "Sid", str(sid))
    check("Sid has a demo appointment to cancel or move", len(sid["upcoming_appointments"]) == 1, str(sid))
    check("Sid does not verify with another patient's date of birth",
          Call().use("verify_patient", said=["Sid", "9th January 2001"], full_name="Sid", date_of_birth="2001-01-09")["verified"] is False)
    # An existing database from before Sid was added gets topped up, with no duplicates.
    with db.connect() as conn:
        conn.execute("DELETE FROM appointments WHERE patient_id IN (SELECT id FROM patients WHERE full_name = 'Sid')")
        conn.execute("DELETE FROM patients WHERE full_name = 'Sid'")
        before = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    db.init_db()
    db.init_db()
    with db.connect() as conn:
        after = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        sids = conn.execute("SELECT COUNT(*) FROM patients WHERE full_name = 'Sid'").fetchone()[0]
        priya = conn.execute("SELECT COUNT(*) FROM appointments WHERE patient_id = (SELECT id FROM patients WHERE full_name = 'Priya Sharma') AND status = 'booked'").fetchone()[0]
    check("a missing demo patient is added on startup, once", after == before + 1 and sids == 1)
    check("existing patients' appointments are not duplicated", priya >= 1 and priya == len(Call().use("verify_patient", said=["Priya Sharma", "14 March 1985"], full_name="Priya Sharma", date_of_birth="1985-03-14")["upcoming_appointments"]))


def _requests(fragment: str) -> list:
    return [c for c in db.staff_snapshot()["callbacks"] if fragment in c["reason"]]


def test_replay_of_a_real_call():
    print("\nReplay of the real call: ticks, Sid, then 'John Curtis'")
    call = Call()
    why = "Patient reporting involuntary movements triggered by sleep deprivation (replay)."
    first = call.use("escalate_to_human", reason=why, urgency="routine")
    check("a clinical question is escalated", first["recorded"] and first["contact_number_on_record"] is False)
    rows = _requests("sleep deprivation (replay)")
    check("it is recorded once, with no caller identified yet", len(rows) == 1 and rows[0]["patient_name"] is None)
    sid = call.use("verify_patient", said=["I'm Sid", "my date of birth is ninth May two thousand one"],
                   full_name="Sid", date_of_birth="2001-05-09")
    check("Sid verifies", sid["verified"] is True)
    rows = _requests("sleep deprivation (replay)")
    check("the earlier escalation now names Sid and has the number on file",
          len(rows) == 1 and rows[0]["patient_name"] == "Sid" and rows[0]["phone_number"] == "07700 900106", str(rows))
    john = call.use("verify_patient", said=["Sid", "John Curtis", "nineteenth October two thousand", "four"],
                    full_name="John Curtis", date_of_birth="2004-10-19")
    check("asking to verify someone else is refused", john["verified"] is False and john["error"] == "already_verified_as_someone_else", str(john))
    check("the refusal does not say whether John is registered", "registered" in john["message"] and "do not say whether" in john["message"])
    check("the call stays verified as Sid, not John", call.state.patient["full_name"] == "Sid")
    check("John gets no access to Sid's appointments",
          "upcoming_appointments" not in john)
    check("booking still acts for Sid only",
          call.use("book_appointment", date=weekday(6).isoformat(), time="14:00", appointment_type="gp", reason="x")["booked"] is True)
    again = call.use("verify_patient", said=["Sid"], full_name="sid", date_of_birth="2001-05-09")
    check("the same person re-verifying is fine", again["verified"] is True and again.get("already_verified") is True)
    sloppy = call.use("verify_patient", said=["Sid"], full_name="Sid", date_of_birth="9 May 2001")
    check("the same patient with an oddly formatted date is not mistaken for someone else", sloppy["verified"] is True)
    callback = call.use("request_callback", reason=why, phone_number="8332097524", caller_name="Sid")
    check("the number is added to the existing record", callback["saved"] is True)
    rows = _requests("sleep deprivation (replay)")
    check("still one record, now with the number Sid gave", len(rows) == 1 and rows[0]["phone_number"] == "8332097524", str(rows))

    print("\nOne concern, one record")
    anon = Call()
    anon.use("escalate_to_human", reason="Wants advice about dizziness (merge test)")
    anon.use("request_callback", reason="Prefers morning calls", phone_number="07700 900888", preferred_time="mornings", caller_name="Kim")
    rows = _requests("dizziness (merge test)")
    check("an anonymous escalation and its callback become one record",
          len(rows) == 1 and rows[0]["phone_number"] == "07700900888" and rows[0]["caller_name"] == "Kim"
          and rows[0]["preferred_time"] == "mornings" and "Also: Prefers morning calls" in rows[0]["reason"], str(rows))
    emergency = Call()
    emergency.use("escalate_to_human", reason="Chest pain (merge test)", urgency="emergency")
    emergency.use("request_callback", reason="Wants test results (merge test)", phone_number="07700 900999")
    check("an emergency is never merged into a later routine request",
          len(_requests("Chest pain (merge test)")) == 1 and len(_requests("Wants test results (merge test)")) == 1)
    check("the emergency record was left untouched", _requests("Chest pain (merge test)")[0]["phone_number"] == "")
    plain = Call()
    plain.use("request_callback", reason="Fit note (merge test)", phone_number="07700 900123")
    check("a callback with nothing to merge into still creates its own record", len(_requests("Fit note (merge test)")) == 1)


def test_admin_requests_and_waiting_list():
    print("\nAdmin requests")
    anon = Call()
    check("an admin request needs verification",
          anon.use("submit_admin_request", request_type="fit_note", details="x")["error"] == "not_verified")
    james = Call()
    james.use("verify_patient", full_name="James Carter", date_of_birth="1972-11-02")
    note = james.use("submit_admin_request", request_type="sick note", details="Needs a fit note for two weeks off work (admin test)")
    check("'sick note' is understood as a fit note", note["saved"] is True and note["type"] == "Fit note", str(note))
    check("a fit note is never promised", "Do not promise" in note["message"])
    results = james.use("submit_admin_request", request_type="test results", details="Wants to know if blood results are back (admin test)")
    check("a results enquiry tells the model not to read out or hint at results",
          "Do not read out, guess or hint" in results["message"] and results["type"] == "Test results enquiry")
    change = james.use("submit_admin_request", request_type="change of details", details="New mobile 07700 900321 (admin test)")
    check("a change of details says the record has not been changed", "has not been changed" in change["message"])
    odd = james.use("submit_admin_request", request_type="banana", details="Something unusual (admin test)")
    check("an unknown type is recorded as 'other', not lost", odd["saved"] is True and odd["type"] == "Other request")
    check("an empty request is refused", james.use("submit_admin_request", request_type="fit_note", details="  ")["saved"] is False)
    mine = [a for a in db.staff_snapshot()["admin_requests"] if "(admin test)" in a["details"]]
    check("all four are on the staff desk with the patient's name",
          len(mine) == 4 and all(a["patient_name"] == "James Carter" for a in mine))
    check("the number on file is used when none is given", all(a["phone_number"] == "07700 900102" for a in mine if "321" not in a["details"]))

    print("\nCancellation list")
    check("joining the list needs verification", anon.use("join_cancellation_list", appointment_type="gp")["error"] == "not_verified")
    tom, aisha = Call(), Call()
    tom.use("verify_patient", full_name="Tom Williams", date_of_birth="2001-01-09")
    aisha.use("verify_patient", full_name="Aisha Khan", date_of_birth="1990-07-21")
    joined = tom.use("join_cancellation_list", appointment_type="GP appointment", note="mornings")
    check("Tom joins the GP list", joined["on_list"] is True and joined["already_on_list"] is False, str(joined))
    check("no slot or time is promised", "Do not promise" in joined["message"])
    again = tom.use("join_cancellation_list", appointment_type="gp", note="any day next week")
    check("asking twice does not create a duplicate", again["already_on_list"] is True)
    tom_rows = [w for w in db.staff_snapshot()["waiting_list"] if w["patient_name"] == "Tom Williams" and w["appt_type"] == "gp"]
    check("one entry, with the newest preference", len(tom_rows) == 1 and tom_rows[0]["note"] == "any day next week" and tom_rows[0]["status"] == "waiting")
    check("an unknown type is rejected", tom.use("join_cancellation_list", appointment_type="dentist")["error"] == "unknown_appointment_type")
    day = weekday(8).isoformat()
    booked = aisha.use("book_appointment", date=day, time="14:00", appointment_type="gp", reason="x")
    check("Aisha books a GP slot", booked["booked"] is True)
    check("booking alone does not flag anyone", [w for w in db.staff_snapshot()["waiting_list"] if w["patient_name"] == "Tom Williams"][0]["status"] == "waiting")
    aisha.use("cancel_appointment", booking_ref=booked["booking_ref"])
    flagged = [w for w in db.staff_snapshot()["waiting_list"] if w["patient_name"] == "Tom Williams" and w["appt_type"] == "gp"][0]
    check("when Aisha cancels, Tom is flagged that a slot opened", flagged["status"] == "slot_open" and "2pm" in flagged["slot_note"], str(flagged))
    check("the flagged slot is really free", "14:00" in db.free_slots("gp", dt.date.fromisoformat(day)))
    check("a slot that opened comes first on the staff desk", db.staff_snapshot()["waiting_list"][0]["status"] == "slot_open")
    moved = aisha.use("book_appointment", date=day, time="15:30", appointment_type="gp", reason="x")
    aisha.use("reschedule_appointment", booking_ref=moved["booking_ref"], new_date=weekday(9).isoformat(), new_time="10:30")
    flagged2 = [w for w in db.staff_snapshot()["waiting_list"] if w["patient_name"] == "Tom Williams" and w["appt_type"] == "gp"][0]
    check("moving an appointment also flags the slot it frees", "3:30pm" in flagged2["slot_note"], str(flagged2))
    nurse = tom.use("join_cancellation_list", appointment_type="nurse")
    check("a different appointment type is a separate entry", nurse["already_on_list"] is False)
    own = Call()
    own.use("verify_patient", full_name="Priya Sharma", date_of_birth="1985-03-14")
    own.use("join_cancellation_list", appointment_type="gp")
    mine_ref = own.use("book_appointment", date=weekday(10).isoformat(), time="09:00", appointment_type="gp", reason="x")["booking_ref"]
    own.use("cancel_appointment", booking_ref=mine_ref)
    priya = [w for w in db.staff_snapshot()["waiting_list"] if w["patient_name"] == "Priya Sharma" and w["appt_type"] == "gp"][0]
    check("cancelling your own slot never flags yourself", priya["status"] == "waiting", str(priya))

    print("\nStaff page shows both")
    import staff_desk
    page = staff_desk.render_page(db.staff_snapshot())
    check("page has the admin requests section", "Admin requests" in page and "Fit note" in page and "Test results enquiry" in page)
    check("page has the cancellation list, with the free slot marked", "Cancellation list" in page and "Slot free" in page)
    check("page escapes what patients say", "<script>" not in page)


# ---- real lines from a live call (5 Oct 2026), trimmed to what the latency report reads ----
LIVE_LOG = """\
2026-10-05 16:12:44.511 | DEBUG    | pipecat.processors.aggregators.llm_response_universal:_on_user_turn_stopped:1441 - LLMUserAggregator#0: User stopped speaking (strategy: TurnAnalyzerUserTurnStopStrategy#0)
2026-10-05 16:12:45.196 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 0, completion tokens: 0
2026-10-05 16:12:45.198 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 0.679s
2026-10-05 16:12:46.811 | DEBUG    | pipecat.processors.aggregators.llm_response_universal:_on_user_turn_stopped:1441 - LLMUserAggregator#0: User stopped speaking (strategy: TurnAnalyzerUserTurnStopStrategy#0)
2026-10-05 16:12:51.796 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 4.979s
2026-10-05 16:12:51.804 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 4983, completion tokens: 37
2026-10-05 16:12:51.806 | DEBUG    | pipecat.services.llm_service:_run_function_call:1723 - GoogleLLMService#0 Calling function [verify_patient:call_544211] with arguments {'full_name': 'Sid', 'date_of_birth': '2001-05-09'}
2026-10-05 16:12:56.122 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 4.288s
2026-10-05 16:12:56.254 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 5125, completion tokens: 41
2026-10-05 16:12:57.201 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - ElevenLabsHttpTTSService#0 TTFB: 1.037s
2026-10-05 16:12:57.204 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:process_ttfa_metrics:228 - ElevenLabsHttpTTSService#0 TTFA: 1.096s (0.059s leading silence)
2026-10-05 16:12:57.204 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:712 - Bot started speaking
2026-10-05 16:13:14.397 | DEBUG    | pipecat.processors.aggregators.llm_response_universal:_on_user_turn_stopped:1441 - LLMUserAggregator#0: User stopped speaking (strategy: TurnAnalyzerUserTurnStopStrategy#0)
2026-10-05 16:13:14.914 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 0, completion tokens: 0
2026-10-05 16:13:14.917 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 0.520s
2026-10-05 16:13:15.254 | DEBUG    | pipecat.processors.aggregators.llm_response_universal:_on_user_turn_stopped:1441 - LLMUserAggregator#0: User stopped speaking (strategy: TurnAnalyzerUserTurnStopStrategy#0)
2026-10-05 16:13:21.314 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 6.055s
2026-10-05 16:13:21.754 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 5194, completion tokens: 94
2026-10-05 16:13:22.198 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:process_ttfa_metrics:228 - ElevenLabsHttpTTSService#0 TTFA: 0.576s (0.047s leading silence)
2026-10-05 16:13:22.206 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:712 - Bot started speaking
2026-10-05 16:14:00.461 | DEBUG    | pipecat.processors.aggregators.llm_response_universal:_on_user_turn_stopped:1441 - LLMUserAggregator#0: User stopped speaking (strategy: TurnAnalyzerUserTurnStopStrategy#0)
2026-10-05 16:14:04.406 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 3.942s
2026-10-05 16:14:04.417 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 5296, completion tokens: 35
2026-10-05 16:14:04.421 | DEBUG    | pipecat.services.llm_service:_run_function_call:1723 - GoogleLLMService#0 Calling function [submit_admin_request:call_602336] with arguments {'details': 'Patient is asking for a referral status update.', 'request_type': 'referral'}
2026-10-05 16:14:08.642 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:stop_ttfb_metrics:175 - GoogleLLMService#0 TTFB: 4.196s
2026-10-05 16:14:08.781 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:start_llm_usage_metrics:337 - GoogleLLMService#0 prompt tokens: 5377, completion tokens: 59
2026-10-05 16:14:09.303 | DEBUG    | pipecat.processors.metrics.frame_processor_metrics:process_ttfa_metrics:228 - ElevenLabsHttpTTSService#0 TTFA: 0.622s (0.052s leading silence)
2026-10-05 16:14:09.311 | DEBUG    | pipecat.transports.base_output:_bot_started_speaking:712 - Bot started speaking
"""


def test_latency_report():
    print("\nLatency report, run on the real call log")
    import tempfile
    import latency_report as lr
    turns = lr.build_turns(lr.parse_events(LIVE_LOG))
    check("three turns are found", len(turns) == 3, str(len(turns)))
    check("silence matches the hand calculation (10.4, 7.0, 8.8 seconds)",
          [round(t["silence"], 1) for t in turns] == [10.4, 7.0, 8.9] or [round(t["silence"], 1) for t in turns] == [10.4, 7.0, 8.8],
          str([round(t["silence"], 2) for t in turns]))
    check("interrupted model calls are ignored, real ones are counted",
          [t["llm_calls"] for t in turns] == [[4.979, 4.288], [6.055], [3.942, 4.196]], str([t["llm_calls"] for t in turns]))
    check("the speech service's own TTFB is not mistaken for the model's",
          all(1.037 not in t["llm_calls"] for t in turns))
    check("tool calls are listed per turn", [t["tools"] for t in turns] == [["verify_patient"], [], ["submit_admin_request"]])
    check("prompt size and first-audio time are picked up",
          [t["prompt_tokens"] for t in turns] == [5125, 5194, 5377] and [t["tts_first_audio"] for t in turns] == [1.096, 0.576, 0.622])
    text = lr.report(turns)
    check("the summary says the model is about 90% of the wait", "90%" in text, text.splitlines()[-2] if text else "")
    check("turns with tools are compared with those without", "turns with a tool" in text)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "run.log"
        path.write_bytes(LIVE_LOG.encode("utf-16"))  # what PowerShell's Tee-Object writes
        check("a UTF-16 log from PowerShell is read correctly", len(lr.build_turns(lr.parse_events(lr.read_text(str(path))))) == 3)
    check("an empty log gives a helpful message", "No finished turns" in lr.report([]))


def test_cut_off_guard():
    print("\nNever act on a half-finished sentence")
    check("'update on the' looks cut off", grounding.looks_cut_off("What is the update on the"))
    check("a trailing 'and' looks cut off", grounding.looks_cut_off("Hi Lara, my name is Sid. And"))
    check("finished sentences are not flagged",
          not any(grounding.looks_cut_off(x) for x in ["Three thirty would be fine.", "Yep.", "I need a fit note.", "No thanks", "Book it please"]))
    james = Call()
    james.use("verify_patient", said=["James Carter", "2 November 1972"], full_name="James Carter", date_of_birth="1972-11-02")
    before = len([a for a in db.staff_snapshot()["admin_requests"] if a["patient_name"] == "James Carter"])
    blocked = james.use("submit_admin_request", said=["What is the update on the"], request_type="referral",
                        details="Patient is asking for a referral status update.")
    check("the referral request from the real call is refused", blocked.get("error") == "patient_sentence_cut_off", str(blocked))
    after = len([a for a in db.staff_snapshot()["admin_requests"] if a["patient_name"] == "James Carter"])
    check("and nothing was saved", after == before)
    done = james.use("submit_admin_request", said=["I need a fit note."], request_type="fit_note", details="Needs a fit note (cutoff test)")
    check("the same request is accepted once the sentence is finished", done["saved"] is True)
    cut = ["Yes please book the"]
    day = weekday(11).isoformat()
    check("booking is refused on a cut-off sentence",
          james.use("book_appointment", said=cut, date=day, time="14:00", appointment_type="gp")["error"] == "patient_sentence_cut_off")
    check("cancelling is refused", james.use("cancel_appointment", said=cut, booking_ref="BK-0001")["error"] == "patient_sentence_cut_off")
    check("rescheduling is refused", james.use("reschedule_appointment", said=cut, booking_ref="BK-0001", new_date=day, new_time="14:00")["error"] == "patient_sentence_cut_off")
    check("a prescription request is refused", james.use("request_repeat_prescription", said=["I need my"], medication_name="Ramipril")["error"] == "patient_sentence_cut_off")
    check("joining the cancellation list is refused", james.use("join_cancellation_list", said=["Put me on the"], appointment_type="gp")["error"] == "patient_sentence_cut_off")
    check("taking a message is refused", james.use("request_callback", said=["Can you tell them that"], reason="x", phone_number="07700 900123")["error"] == "patient_sentence_cut_off")
    emergency = Call().use("escalate_to_human", said=["I have chest pain and my"], reason="Chest pain (cutoff test)", urgency="emergency")
    check("an emergency escalation is never held up", emergency["recorded"] is True)
    check("reading tools are never held up", Call().use("search_health_info", said=["How long does a cold"], query="cold")["found"] is True)
    check("with no readable conversation the guard stays out of the way",
          Call().use("submit_admin_request", said=None, request_type="other", details="x") ["error"] == "not_verified")


def test_prompt_modes_and_budget():
    print("\nPrompt options and the token budget")
    import importlib
    import prompts
    saved = os.environ.get("LARA_PROMPT")
    try:
        os.environ["LARA_PROMPT"] = "full"
        full = prompts.build_system_prompt()
        os.environ["LARA_PROMPT"] = "compact"
        compact = prompts.build_system_prompt()
        os.environ.pop("LARA_PROMPT")
        default = prompts.build_system_prompt()
    finally:
        if saved is None:
            os.environ.pop("LARA_PROMPT", None)
        else:
            os.environ["LARA_PROMPT"] = saved
    check("the compact prompt is the default", default == compact)
    check("the compact prompt is at least 20% shorter than the full one", len(compact) < 0.8 * len(full), f"{len(compact)} vs {len(full)}")
    for phrase, why in [("never diagnose", "no diagnosing"), ("999", "emergencies"), ("verify_patient", "verification"),
                        ("escalate_to_human", "escalation"), ("cannot transfer", "no transfers"),
                        ("cut off", "half-finished sentences"), ("never say whether anyone is registered", "no proxy access"),
                        ("search_health_info", "health lookups"), ("Today is", "today's date")]:
        check(f"compact prompt keeps the rule: {why}", phrase in compact)
    try:
        import json
        from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
    except ImportError:
        print("  skipped the token budget (pipecat not installed)")
        return
    schema_chars = 0
    for fn in Call().tools.values():
        sc = DirectFunctionWrapper(fn).to_function_schema()
        schema_chars += len(json.dumps({"name": sc.name, "description": sc.description, "properties": sc.properties, "required": sc.required}))
    total = (len(compact) + schema_chars) // 4
    check(f"fixed prompt plus tool descriptions stay under 3,300 tokens (now about {total})", total < 3300)


def test_fillers():
    print("\n'One moment' while a tool runs")
    try:
        from pipecat.frames.frames import TTSSpeakFrame
        from pipecat.services.google.llm import GoogleLLMService
        from pipecat.services.llm_service import FunctionCallFromLLM
    except ImportError:
        print("  skipped (pipecat not installed)")
        return
    import fillers
    check("slow tools have a filler", all(fillers.pick_filler([n]) for n in
          ["verify_patient", "check_availability", "book_appointment", "cancel_appointment", "reschedule_appointment",
           "request_repeat_prescription", "submit_admin_request", "join_cancellation_list"]))
    check("emergencies and quick lookups never get one", not any(fillers.pick_filler([n]) for n in
          ["escalate_to_human", "search_knowledge_base", "search_health_info", "get_surgery_status", "request_callback"]))
    os.environ.setdefault("GOOGLE_API_KEY", "dummy")
    llm = GoogleLLMService(api_key="dummy", settings=GoogleLLMService.Settings(system_instruction="x"))
    sent = []

    class FakeTTS:
        async def queue_frame(self, frame, *args, **kwargs):
            sent.append(frame)

    fillers.attach_fillers(llm, FakeTTS())

    async def fire(name):
        sent.clear()
        await llm._call_event_handler("on_function_calls_started",
                                      [FunctionCallFromLLM(function_name=name, tool_call_id="1", arguments={}, context=None)])
        await asyncio.sleep(0.2)  # Pipecat runs event handlers as background tasks
        return [f.text for f in sent if isinstance(f, TTSSpeakFrame)]

    spoken = asyncio.run(fire("verify_patient"))
    check("a booking-type tool queues a spoken phrase", len(spoken) == 1 and spoken[0] in fillers.FILLERS["verify_patient"], str(spoken))
    check("escalation stays silent", asyncio.run(fire("escalate_to_human")) == [])


def test_booking():
    print("\nBooking, cancelling, rescheduling")
    day = weekday(3).isoformat()
    a, b = Call(), Call()
    a.use("verify_patient", full_name="Aisha Khan", date_of_birth="1990-07-21")
    b.use("verify_patient", full_name="Tom Williams", date_of_birth="2001-01-09")
    avail = a.use("check_availability", date=day, appointment_type="gp")
    check("availability lists GP times", [t["time"] for t in avail["available_times"]] == db.BASE_SLOTS["gp"])
    booked = a.use("book_appointment", date=day, time="14:00", appointment_type="gp", reason="check-up")
    check("books a free slot", booked["booked"] and booked["time"] == "14:00", str(booked))
    check("spoken form is friendly", booked["spoken_when"].endswith("at 2pm"))
    check("booked slot disappears from availability",
          "14:00" not in [t["time"] for t in a.use("check_availability", date=day)["available_times"]])
    taken = b.use("book_appointment", date=day, time="14:00", appointment_type="gp")
    check("someone else cannot take the same slot", taken["booked"] is False and taken["error"] == "slot_unavailable")
    check("a refusal offers alternatives", len(taken["next_available"]) > 0)
    first = a.use("book_appointment", date=day, time="09:00", appointment_type="gp")
    clash = a.use("book_appointment", date=day, time="09:00", appointment_type="blood_test")
    check("same patient cannot be in two places at once", clash["booked"] is False and clash["error"] == "patient_clash", str(clash))
    a.use("cancel_appointment", booking_ref=first["booking_ref"])
    check("weekend has no slots", next_saturday() and not db.free_slots("gp", next_saturday()))
    check("past dates have no slots", not db.free_slots("gp", clock.uk_today() - dt.timedelta(days=1)))
    check("cannot book more than three weeks ahead",
          not db.free_slots("gp", clock.uk_today() + dt.timedelta(days=40)))
    ref = booked["booking_ref"]
    check("another patient cannot cancel it", b.use("cancel_appointment", booking_ref=ref)["cancelled"] is False)
    check("another patient cannot move it",
          b.use("reschedule_appointment", booking_ref=ref, new_date=day, new_time="15:30")["rescheduled"] is False)
    moved = a.use("reschedule_appointment", booking_ref=ref, new_date=day, new_time="15:30")
    check("owner can reschedule", moved["rescheduled"] and moved["now"]["time"] == "15:30", str(moved))
    check("old slot is free again", "14:00" in [t["time"] for t in a.use("check_availability", date=day)["available_times"]])
    blocked = a.use("reschedule_appointment", booking_ref=ref, new_date=day, new_time="not a time")
    check("bad time is rejected cleanly", blocked["rescheduled"] is False)
    cancelled = a.use("cancel_appointment", booking_ref=ref)
    check("owner can cancel", cancelled["cancelled"] is True)
    check("cancelling twice is refused", a.use("cancel_appointment", booking_ref=ref)["cancelled"] is False)
    check("cancelled slot can be booked by someone else",
          b.use("book_appointment", date=day, time="15:30", appointment_type="gp")["booked"] is True)
    check("unknown booking reference is refused", a.use("cancel_appointment", booking_ref="BK-9999")["cancelled"] is False)
    check("nurse and blood test slots work",
          a.use("book_appointment", date=day, time="09:30", appointment_type="nurse")["booked"]
          and a.use("book_appointment", date=day, time="08:20", appointment_type="blood tests")["booked"])


def test_today_cutoff():
    print("\nSlots that have already passed today")
    monday_noon = dt.datetime(2026, 10, 5, 12, 0)
    times = db.free_slots("gp", monday_noon.date(), now=monday_noon)
    check("morning slots are gone at noon", times == ["14:00", "15:30"], str(times))
    check("needs 15 minutes' notice", "14:00" not in db.free_slots("gp", monday_noon.date(), now=dt.datetime(2026, 10, 5, 13, 50)))


def test_prescriptions_and_messages():
    print("\nPrescriptions, callbacks, escalation")
    call = Call()
    call.use("verify_patient", full_name="James Carter", date_of_birth="1972-11-02")
    check("repeat prescription is recorded",
          call.use("request_repeat_prescription", medication_name="Ramipril")["requested"] is True)
    anon = Call()
    check("callback needs a phone number when not verified",
          anon.use("request_callback", reason="test results")["error"] == "need_phone_number")
    check("callback rejects an impossible number",
          anon.use("request_callback", reason="test results", phone_number="123")["saved"] is False)
    check("callback saved with a valid number",
          anon.use("request_callback", reason="test results", phone_number="07700 900555", caller_name="Sam")["saved"] is True)
    check("verified patient needs no number given",
          call.use("request_callback", reason="fit note")["saved"] is True)
    emergency = anon.use("escalate_to_human", reason="chest pain", urgency="emergency")
    check("emergency escalation records immediately with no phone number",
          emergency["recorded"] and emergency["urgency"] == "emergency")
    routine = Call().use("escalate_to_human", reason="wants a diagnosis")
    check("anonymous routine escalation asks for a number", "request_callback" in routine["message"])
    check("odd urgency falls back to routine",
          Call().use("escalate_to_human", reason="x", urgency="banana")["urgency"] == "routine")
    check("escalation tells the model it cannot transfer calls", "cannot transfer" in routine["reminder"])


def test_status_and_search():
    print("\nOpening hours tool and knowledge search")
    call = Call()
    check("saturday is closed", call.use("get_surgery_status", day=next_saturday().isoformat())["open"] is False)
    check("a weekday has hours", call.use("get_surgery_status", day=weekday(1).isoformat())["open"] is True)
    check("today gives live status", "open_now" in call.use("get_surgery_status", day="today"))
    check("nonsense day is handled", call.use("get_surgery_status", day="blorp")["error"] == "bad_day")
    check("health search finds a cold", call.use("search_health_info", query="how long does a cold last")["found"])
    check("health search will not answer medication questions",
          call.use("search_health_info", query="can I take ibuprofen for my knee")["found"] is False)
    check("faq search works", call.use("search_knowledge_base", query="do you have parking")["found"])


def test_staff_snapshot():
    print("\nStaff desk data")
    snap = db.staff_snapshot()
    check("emergencies sort first",
          snap["callbacks"] and snap["callbacks"][0]["urgency"] == "emergency")
    check("appointments carry patient names", all(a["patient_name"] for a in snap["appointments"]))
    check("prescription requests are listed", len(snap["prescriptions"]) >= 1)


def test_call_summary():
    print("\nPost-call summary")
    try:
        from pipecat.processors.aggregators.llm_context import LLMContext
    except ImportError:
        print("  skipped (pipecat not installed)")
        return
    import call_summary as cs

    call = Call()
    call.use("verify_patient", full_name="Aisha Khan", date_of_birth="1990-07-21")
    booked = call.use("book_appointment", date=weekday(5).isoformat(), time="10:30", appointment_type="gp", reason="check-up")
    call.use("escalate_to_human", reason="asked for a diagnosis", urgency="urgent")
    context = LLMContext()
    context.add_message({"role": "developer", "content": "Greet the patient."})
    context.add_message({"role": "assistant", "content": "Hello, you have reached Riverside Surgery."})
    context.add_message({"role": "user", "content": "I'd like to book an appointment."})
    context.add_message({"role": "user", "content": [{"type": "text", "text": "Aisha Khan, 21 July 1990."}]})
    lines = cs.transcript_from_context(context)
    check("transcript keeps patient and Lara turns only", len(lines) == 3 and lines[0].startswith("Lara:"), str(lines))
    check("transcript handles list-style content", lines[2] == "Patient: Aisha Khan, 21 July 1990.")
    fallback = cs.fallback_summary(call.state, lines)
    check("fallback records the booking and escalation from what really happened",
          fallback["follow_up_needed"] and fallback["urgency"] == "urgent" and fallback["patient_verified"]
          and any("booked" in a for a in fallback["actions_taken"]), str(fallback))
    messy = '```json\n{"reason_for_call": "Book a visit", "urgency": "banana", "actions_taken": ["Booked GP"], "patient_verified": false}\n```'
    tidy = cs._tidy(cs._parse_json(messy), call.state)
    check("code-fenced JSON from the model is understood", tidy["reason_for_call"] == "Book a visit")
    check("an invalid urgency is corrected", tidy["urgency"] == "none")
    check("verified status comes from what happened, not what the model says", tidy["patient_verified"] is True)
    check("not-JSON from the model is rejected", cs._parse_json("sorry, I can't") is None)
    asyncio.run(cs.finish_call(context, call.state, api_key=None, model="x"))
    saved = db.staff_snapshot()["summaries"]
    check("a summary is saved even with no API key", saved and saved[0]["summary"]["source"] == "fallback")
    asyncio.run(cs.finish_call(context, call.state, api_key="not-a-real-key", model="gemini-3.1-flash-lite"))
    check("a failing Gemini call falls back instead of crashing", len(db.staff_snapshot()["summaries"]) >= 2)
    silent = LLMContext()
    silent.add_message({"role": "assistant", "content": "Hello."})
    before = len(db.staff_snapshot()["summaries"])
    asyncio.run(cs.finish_call(silent, Call().state, api_key=None, model="x"))
    check("a call where the patient never spoke is not summarised", len(db.staff_snapshot()["summaries"]) == before)


def test_pipecat_accepts_the_tools():
    print("\nPipecat accepts every tool")
    try:
        from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
    except ImportError:
        print("  skipped (pipecat not installed)")
        return
    for fn in Call().tools.values():
        try:
            wrapper = DirectFunctionWrapper(fn)
            schema = wrapper.to_function_schema()
            props = list(schema.properties)
            check(f"{fn.__name__}({', '.join(props)}); required={list(schema.required)}",
                  bool(schema.description) and bool(props))
        except Exception as error:  # noqa: BLE001
            check(f"{fn.__name__} accepted by Pipecat", False, str(error))


if __name__ == "__main__":
    db.init_db()
    for test in (test_clock, test_parsing, test_verification, test_grounding, test_demo_patients, test_replay_of_a_real_call, test_admin_requests_and_waiting_list, test_latency_report, test_cut_off_guard, test_prompt_modes_and_budget, test_fillers, test_booking, test_today_cutoff,
                 test_prescriptions_and_messages, test_status_and_search, test_call_summary, test_staff_snapshot,
                 test_pipecat_accepts_the_tools):
        test()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed:", *FAILED, sep="\n  - ")
        sys.exit(1)