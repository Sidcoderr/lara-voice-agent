"""Tools Lara can call.

Pipecat "direct functions": the schema the model sees is inferred from each
function's signature and docstring. The first parameter must be named `params`.

Tools are built once per call by build_tools(), so every call has its own
SessionState (who has been verified, failed attempts, what happened). The
rules that matter for safety are enforced here in code, not just in the
prompt: booking, changing, cancelling and prescription requests all refuse to
run until verify_patient has succeeded.
"""

import re
import uuid
from dataclasses import dataclass, field

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

import grounding
import surgery_clock as clock
import surgery_db as db
from knowledge_base import health_kb, kb

MAX_VERIFY_ATTEMPTS = 3
URGENCY_LEVELS = ("routine", "urgent", "emergency")


@dataclass
class SessionState:
    """Everything we remember about one call."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    patient: dict | None = None
    failed_verifications: int = 0
    events: list = field(default_factory=list)
    # Escalations recorded during this call, so later details can be added to them
    # instead of creating a second record for the same concern.
    escalation_ids: list = field(default_factory=list)
    awaiting_contact: list = field(default_factory=list)  # non-emergency, no number given yet
    # The emergency record for this call, if there is one, and which kinds of emergency the
    # safety net has already spoken about. A second emergency report adds to the same record.
    emergency_id: int | None = None
    alerted: set = field(default_factory=set)

    def log(self, tool: str, outcome: str, **details) -> None:
        self.events.append({"tool": tool, "outcome": outcome, **details})
        logger.info(f"[{self.session_id}] {tool}: {outcome} {details or ''}")


def _clean_phone(text: str) -> str | None:
    """Keep digits and a leading +; accept 10 to 13 digits."""
    cleaned = re.sub(r"[^\d+]", "", text or "")
    digits = cleaned.replace("+", "")
    return cleaned if 10 <= len(digits) <= 13 else None


def build_tools(state: SessionState) -> list:
    """Create this call's tools. Register each with llm.register_direct_function."""
    db.init_db()

    def not_verified() -> dict:
        return {
            "error": "not_verified",
            "message": (
                "This patient has not been verified. Ask for their full name and date "
                "of birth and call verify_patient first."
            ),
        }

    def cut_off(params) -> bool:
        """Was the patient's last sentence cut off? Then nothing should be changed on it."""
        last = grounding.last_patient_message(getattr(params, "context", None))
        return last is not None and grounding.looks_cut_off(last)

    async def ask_to_finish(params, tool: str) -> None:
        state.log(tool, "not run: the patient's sentence seemed cut off")
        await params.result_callback(
            {
                "error": "patient_sentence_cut_off",
                "message": (
                    "The patient's last sentence seems to have been cut off, so nothing has "
                    "been done. Ask them to finish what they were saying."
                ),
            }
        )

    def bad_type(text: str) -> dict:
        return {
            "error": "unknown_appointment_type",
            "message": f"'{text}' is not an appointment type.",
            "valid_types": list(db.APPT_TYPES),
        }

    # ------------------------------------------------------------------ information

    async def search_knowledge_base(params: FunctionCallParams, query: str):
        """Look up surgery information: registering, what to bring, blood test prep, cancellation policy, prescription timing, phone or video appointments, interpreters, parking. Not for health questions.

        Args:
            query: The patient's question.
        """
        results = kb.search(query)
        state.log("search_knowledge_base", f"{len(results)} result(s)", query=query)
        if not results:
            await params.result_callback(
                {"found": False, "message": "No matching information in the knowledge base."}
            )
            return
        await params.result_callback({"found": True, "results": results})

    async def search_health_info(params: FunctionCallParams, query: str):
        """Look up general health information and signposting: colds, sore throat, headache, fever, cough, sickness and diarrhoea, back pain, and when to call 999 or 111. General information only; never use it to diagnose.

        Args:
            query: The health topic, in the patient's words.
        """
        results = health_kb.search(query, top_k=2, min_score=3)
        state.log("search_health_info", f"{len(results)} result(s)", query=query)
        if not results:
            await params.result_callback(
                {
                    "found": False,
                    "message": (
                        "No reliable information available on this. Do not guess. "
                        "Offer a GP appointment or a call back from a clinician."
                    ),
                }
            )
            return
        await params.result_callback(
            {
                "found": True,
                "results": results,
                "reminder": "General information only, not medical advice. Do not diagnose.",
            }
        )

    async def get_surgery_status(params: FunctionCallParams, day: str = "today"):
        """Check whether the surgery is open, now or on a given day.

        Args:
            day: 'today', 'tomorrow', a weekday name, or YYYY-MM-DD.
        """
        target = clock.resolve_day(day)
        if target is None:
            await params.result_callback(
                {"error": "bad_day", "message": "Could not understand that day. Ask again."}
            )
            return
        if target == clock.uk_today():
            payload = clock.surgery_status()
        else:
            payload = clock.day_hours(target)
        state.log("get_surgery_status", "ok", day=day)
        await params.result_callback(payload)

    # ------------------------------------------------------------------- appointments

    async def check_availability(
        params: FunctionCallParams, date: str, appointment_type: str = "gp"
    ):
        """Check free appointment times on a date. Offer two or three.

        Args:
            date: YYYY-MM-DD.
            appointment_type: gp, nurse, phone_consultation or blood_test.
        """
        appt_type = db.normalise_type(appointment_type)
        if appt_type is None:
            await params.result_callback(bad_type(appointment_type))
            return
        day = db.parse_date(date)
        if day is None:
            await params.result_callback(
                {"error": "bad_date", "message": "The date must be in YYYY-MM-DD format."}
            )
            return
        times = db.free_slots(appt_type, day)
        state.log("check_availability", f"{len(times)} free", type=appt_type, date=date)
        if times:
            await params.result_callback(
                {
                    "date": clock.speak_date(day),
                    "appointment_type": db.APPT_TYPES[appt_type],
                    "available_times": [
                        {"time": t, "say": clock.format_time(t)} for t in times
                    ],
                }
            )
            return
        await params.result_callback(
            {
                "date": clock.speak_date(day),
                "appointment_type": db.APPT_TYPES[appt_type],
                "available_times": [],
                "message": (
                    "Nothing free that day. The surgery is closed at weekends, and "
                    "dates in the past or more than three weeks ahead cannot be booked."
                ),
                "next_available": db.next_available(appt_type, day),
            }
        )

    async def verify_patient(params: FunctionCallParams, full_name: str, date_of_birth: str):
        """Verify the patient. Call only after they have said their full name and a complete date of birth (day, month, year) in this call; never guess missing parts.

        Args:
            full_name: Their first and last name, as said.
            date_of_birth: The date of birth they said, as YYYY-MM-DD.
        """
        if state.patient:
            first_name = state.patient["full_name"].split()[0]
            if db.is_same_person(state.patient, full_name, db.parse_date(date_of_birth)):
                await params.result_callback(
                    {
                        "verified": True,
                        "already_verified": True,
                        "first_name": first_name,
                        "upcoming_appointments": db.upcoming_appointments(state.patient["id"]),
                    }
                )
                return
            # Someone else's details. Never confirm or deny that they are a patient here.
            state.log("verify_patient", "refused: details are for someone else")
            await params.result_callback(
                {
                    "verified": False,
                    "error": "already_verified_as_someone_else",
                    "message": (
                        f"This call has already been verified as {first_name}, and you can "
                        "only help that person. Do not verify or take details for anyone "
                        "else, and do not say whether anyone else is registered here. "
                        "Explain that the other person needs to contact the surgery "
                        "themselves."
                    ),
                }
            )
            return
        if state.failed_verifications >= MAX_VERIFY_ATTEMPTS:
            await params.result_callback(
                {
                    "verified": False,
                    "locked": True,
                    "message": (
                        "Too many failed attempts. Do not try again on this call. Offer "
                        "to take a callback request with request_callback."
                    ),
                }
            )
            return
        dob = db.parse_date(date_of_birth)
        if dob is None:
            await params.result_callback(
                {
                    "verified": False,
                    "error": "bad_date_format",
                    "message": "Ask the patient to repeat their date of birth, then pass it as YYYY-MM-DD.",
                }
            )
            return
        # Do not take the model's word for it: the patient must really have said
        # this date of birth. (If the conversation cannot be read, this is skipped.)
        said = grounding.recent_patient_words(getattr(params, "context", None))
        if said is None:
            logger.warning(
                f"[{state.session_id}] Could not read the conversation, so the date of birth "
                "was not checked against what the patient said."
            )
        if said is not None and not grounding.dob_was_said(dob, said):
            state.log("verify_patient", "date of birth not said by the patient")
            await params.result_callback(
                {
                    "verified": False,
                    "error": "date_of_birth_not_heard",
                    "message": (
                        "The patient has not said a full date of birth (day, month and year) "
                        "in this call, so do not guess one. Ask them for it, then call "
                        "verify_patient again. This does not count as a failed attempt."
                    ),
                }
            )
            return
        patient = db.find_patient(full_name, dob)
        if patient is None:
            state.failed_verifications += 1
            state.log("verify_patient", "no match", attempt=state.failed_verifications)
            await params.result_callback(
                {
                    "verified": False,
                    "attempts_left": MAX_VERIFY_ATTEMPTS - state.failed_verifications,
                    "message": (
                        "Those details did not match a patient record. Do not say which "
                        "detail was wrong. Ask them to repeat their full name (you may ask them to "
                        "spell it) and date of birth, or offer a callback if no attempts are left."
                    ),
                }
            )
            return
        state.patient = patient
        state.log("verify_patient", "verified", patient_id=patient["id"])
        db.attach_callback_details(
            state.escalation_ids, patient_id=patient["id"],
            caller_name=patient["full_name"], phone_if_missing=patient["phone"],
        )
        await params.result_callback(
            {
                "verified": True,
                "first_name": patient["full_name"].split()[0],
                "upcoming_appointments": db.upcoming_appointments(patient["id"]),
            }
        )

    async def book_appointment(
        params: FunctionCallParams,
        date: str,
        time: str,
        appointment_type: str = "gp",
        reason: str = "",
    ):
        """Book an appointment for the verified patient, after checking availability and getting their confirmation.

        Args:
            date: YYYY-MM-DD.
            time: 24-hour HH:MM, such as 14:30.
            appointment_type: gp, nurse, phone_consultation or blood_test.
            reason: A brief non-clinical reason, such as "follow-up".
        """
        if cut_off(params):
            await ask_to_finish(params, 'book_appointment')
            return
        if state.patient is None:
            await params.result_callback(not_verified())
            return
        appt_type = db.normalise_type(appointment_type)
        if appt_type is None:
            await params.result_callback(bad_type(appointment_type))
            return
        day, time_text = db.parse_date(date), db.parse_time(time)
        if day is None or time_text is None:
            await params.result_callback(
                {"error": "bad_date_or_time", "message": "Use YYYY-MM-DD for the date and HH:MM for the time."}
            )
            return
        result = db.book(state.patient["id"], appt_type, day, time_text, reason)
        if result["ok"]:
            state.log("book_appointment", "booked", **{k: result["appointment"][k] for k in ("booking_ref", "spoken_when", "type")})
            await params.result_callback({"booked": True, **result["appointment"]})
        else:
            state.log("book_appointment", result["error"])
            await params.result_callback({"booked": False, **{k: v for k, v in result.items() if k != "ok"}})

    async def cancel_appointment(params: FunctionCallParams, booking_ref: str):
        """Cancel one of the verified patient's upcoming appointments, after confirming with them.

        Args:
            booking_ref: The reference from verify_patient or book_appointment, such as BK-0007.
        """
        if cut_off(params):
            await ask_to_finish(params, 'cancel_appointment')
            return
        if state.patient is None:
            await params.result_callback(not_verified())
            return
        appt_id = db.parse_booking_ref(booking_ref)
        if appt_id is None:
            await params.result_callback({"cancelled": False, "error": "bad_reference"})
            return
        result = db.cancel(state.patient["id"], appt_id)
        if result["ok"]:
            state.log("cancel_appointment", "cancelled", booking_ref=result["cancelled"]["booking_ref"])
            await params.result_callback({"cancelled": True, **result["cancelled"]})
        else:
            state.log("cancel_appointment", result["error"])
            await params.result_callback({"cancelled": False, **{k: v for k, v in result.items() if k != "ok"}})

    async def reschedule_appointment(
        params: FunctionCallParams, booking_ref: str, new_date: str, new_time: str
    ):
        """Move an upcoming appointment of the verified patient to a new slot of the same type. Check the slot first and confirm with them.

        Args:
            booking_ref: The reference, such as BK-0007.
            new_date: YYYY-MM-DD.
            new_time: 24-hour HH:MM.
        """
        if cut_off(params):
            await ask_to_finish(params, 'reschedule_appointment')
            return
        if state.patient is None:
            await params.result_callback(not_verified())
            return
        appt_id = db.parse_booking_ref(booking_ref)
        day, time_text = db.parse_date(new_date), db.parse_time(new_time)
        if appt_id is None or day is None or time_text is None:
            await params.result_callback(
                {"rescheduled": False, "error": "bad_input",
                 "message": "Need a booking reference, a YYYY-MM-DD date and an HH:MM time."}
            )
            return
        result = db.reschedule(state.patient["id"], appt_id, day, time_text)
        if result["ok"]:
            state.log("reschedule_appointment", "moved", booking_ref=result["appointment"]["booking_ref"],
                      spoken_when=result["appointment"]["spoken_when"])
            await params.result_callback(
                {"rescheduled": True, "previous": result["previous"], "now": result["appointment"]}
            )
        else:
            state.log("reschedule_appointment", result["error"])
            await params.result_callback({"rescheduled": False, **{k: v for k, v in result.items() if k != "ok"}})

    async def request_repeat_prescription(params: FunctionCallParams, medication_name: str):
        """Request a repeat of a medicine the verified patient already takes. Not for new medicines or dose changes. Confirm the name first.

        Args:
            medication_name: The medicine's name.
        """
        if cut_off(params):
            await ask_to_finish(params, 'request_repeat_prescription')
            return
        if state.patient is None:
            await params.result_callback(not_verified())
            return
        name = (medication_name or "").strip()
        if not name:
            await params.result_callback(
                {"requested": False, "message": "Ask which medicine they need."}
            )
            return
        request_id = db.add_prescription_request(state.patient["id"], name)
        state.log("request_repeat_prescription", "requested", medication=name, request_id=request_id)
        await params.result_callback(
            {
                "requested": True,
                "medication": name,
                "message": "Submitted. Please allow 2 working days for it to be ready at the pharmacy.",
            }
        )

    # ----------------------------------------------- admin requests and the waiting list

    async def submit_admin_request(
        params: FunctionCallParams, request_type: str, details: str, phone_number: str = ""
    ):
        """Record an admin request for the verified patient. Never read out or hint at test results; never promise a fit note, a time or a fee.

        Args:
            request_type: fit_note, test_results, referral, letter_or_form, change_of_details or other.
            details: What they asked for, including any new details given.
            phone_number: Only if different from the number on record.
        """
        if cut_off(params):
            await ask_to_finish(params, 'submit_admin_request')
            return
        if state.patient is None:
            await params.result_callback(not_verified())
            return
        text = (details or "").strip()
        if not text:
            await params.result_callback(
                {"saved": False, "message": "Ask what exactly they need, then try again."}
            )
            return
        kind = db.normalise_admin_type(request_type)
        number = _clean_phone(phone_number) or state.patient["phone"]
        request_id = db.add_admin_request(state.patient["id"], kind, text, number)
        state.log("submit_admin_request", "saved", request_type=kind, request_id=request_id)
        say = {
            "fit_note": "A clinician has to approve fit notes, so the team will review it and be in touch. Do not promise one will be issued.",
            "test_results": "Do not read out, guess or hint at any results. The team or a clinician will be in touch.",
            "referral": "The team will check the referral and be in touch.",
            "letter_or_form": "The team will review it and be in touch. Do not promise a time or a fee.",
            "change_of_details": "Staff will confirm the new details before changing the record. The record has not been changed yet.",
            "other": "The team will review it and be in touch.",
        }[kind]
        await params.result_callback(
            {"saved": True, "type": db.ADMIN_TYPES[kind], "message": f"Recorded. {say} Do not promise a time."}
        )

    async def join_cancellation_list(
        params: FunctionCallParams, appointment_type: str = "gp", note: str = ""
    ):
        """Add the verified patient to the cancellation list for an appointment type when nothing suits. Never promise a slot.

        Args:
            appointment_type: gp, nurse, phone_consultation or blood_test.
            note: Their preferences, such as "mornings".
        """
        if cut_off(params):
            await ask_to_finish(params, 'join_cancellation_list')
            return
        if state.patient is None:
            await params.result_callback(not_verified())
            return
        appt_type = db.normalise_type(appointment_type)
        if appt_type is None:
            await params.result_callback(bad_type(appointment_type))
            return
        _, added = db.join_waiting_list(state.patient["id"], appt_type, note)
        state.log("join_cancellation_list", "added" if added else "updated", type=appt_type)
        await params.result_callback(
            {
                "on_list": True,
                "appointment_type": db.APPT_TYPES[appt_type],
                "already_on_list": not added,
                "message": "The surgery will contact them if a suitable slot opens. Do not promise a slot or a time.",
            }
        )

    # ------------------------------------------------------- messages and escalation

    async def request_callback(
        params: FunctionCallParams,
        reason: str,
        phone_number: str = "",
        preferred_time: str = "",
        caller_name: str = "",
    ):
        """Take a message so the team can ring the patient back, for non-clinical things you cannot do, or when they cannot be verified. For clinical or urgent matters use escalate_to_human.

        Args:
            reason: What they need, in a sentence.
            phone_number: Number to call back; not needed if verified.
            preferred_time: When they would like the call, if said.
            caller_name: The caller's name, if given.
        """
        if cut_off(params):
            await ask_to_finish(params, 'request_callback')
            return
        number = _clean_phone(phone_number)
        if number is None and state.patient:
            number = state.patient["phone"]
        if number is None:
            await params.result_callback(
                {"saved": False, "error": "need_phone_number",
                 "message": "Ask for a phone number to call back on, and read it back to confirm."}
            )
            return
        name = caller_name or (state.patient["full_name"] if state.patient else "")
        if state.awaiting_contact:
            # This is the number for the concern already flagged: update that one
            # record rather than creating a second one.
            db.attach_callback_details(
                state.awaiting_contact,
                patient_id=state.patient["id"] if state.patient else None,
                caller_name=name, phone_number=number, preferred_time=preferred_time,
                extra_reason=reason,
            )
            state.log("request_callback", "added to the earlier escalation",
                      callback_ids=list(state.awaiting_contact))
            state.awaiting_contact.clear()
            await params.result_callback(
                {
                    "saved": True,
                    "message": "Number recorded with the request already flagged. The team will ring them back on it. Do not promise a time.",
                }
            )
            return
        callback_id = db.add_callback(
            "callback", "routine", reason,
            patient_id=state.patient["id"] if state.patient else None,
            caller_name=name, phone_number=number, preferred_time=preferred_time,
        )
        state.log("request_callback", "saved", callback_id=callback_id)
        await params.result_callback(
            {
                "saved": True,
                "message": "Message recorded. The team will ring them back on that number. Do not promise a time.",
            }
        )

    async def escalate_to_human(
        params: FunctionCallParams, reason: str, urgency: str = "routine", phone_number: str = ""
    ):
        """Flag this call for a clinician or staff. It does not transfer the call. Use for diagnosis or medical advice requests, medication help, babies, young children, pregnancy, very unwell callers, distress, requests for a person, and emergency signs (also tell them to call 999).

        Args:
            reason: A short summary for the staff member.
            urgency: emergency, urgent or routine.
            phone_number: Their number if given. Do not ask in an emergency.
        """
        level = (urgency or "routine").strip().lower()
        if level not in URGENCY_LEVELS:
            level = "routine"
        number = _clean_phone(phone_number) or (state.patient["phone"] if state.patient else "")
        if level == "emergency" and state.emergency_id is not None:
            # The safety net (or an earlier report) already raised this emergency: add to it.
            callback_id = state.emergency_id
            db.attach_callback_details(
                [callback_id], patient_id=state.patient["id"] if state.patient else None,
                caller_name=state.patient["full_name"] if state.patient else "",
                phone_if_missing=number or "", extra_reason=reason,
            )
        else:
            callback_id = db.add_callback(
                "escalation", level, reason,
                patient_id=state.patient["id"] if state.patient else None,
                caller_name=state.patient["full_name"] if state.patient else "",
                phone_number=number or "",
            )
            state.escalation_ids.append(callback_id)
            if level == "emergency":
                state.emergency_id = callback_id
            elif not number:
                state.awaiting_contact.append(callback_id)
        state.log("escalate_to_human", level, callback_id=callback_id, reason=reason)
        advice = {
            "emergency": "Recorded for staff. Make sure the patient has been told to call 999 now.",
            "urgent": "Recorded as urgent. Tell them a clinician will review it, and that for advice right now they can call NHS 111.",
            "routine": "Recorded for a clinician to review.",
        }[level]
        await params.result_callback(
            {
                "recorded": True,
                "urgency": level,
                "contact_number_on_record": bool(number),
                "message": advice + (
                    "" if number or level == "emergency"
                    else " No contact number is on record: ask for one and call request_callback."
                ),
                "reminder": "You cannot transfer the call or put anyone on hold. Do not say a colleague is joining.",
            }
        )

    return [
        search_knowledge_base,
        search_health_info,
        get_surgery_status,
        check_availability,
        verify_patient,
        book_appointment,
        cancel_appointment,
        reschedule_appointment,
        request_repeat_prescription,
        submit_admin_request,
        join_cancellation_list,
        request_callback,
        escalate_to_human,
    ]