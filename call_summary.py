"""End-of-call summary for the staff desk.

When a call ends we read the conversation back from the context, ask Gemini for
a short structured summary, and save it. If anything goes wrong (no network, a
rate limit, bad JSON) we fall back to a plain record built from what the tools
actually did, and we never let a failure here break the end of a call.
"""

import asyncio
import json
import re

from loguru import logger

import surgery_db as db

SUMMARY_TIMEOUT_SECONDS = 25

_SUMMARY_INSTRUCTIONS = """You are writing a note for the reception team at Riverside Surgery, an NHS GP surgery, about a phone call between a patient and Lara, the surgery's AI receptionist.

Use ONLY what is in the transcript and the list of actions. Do not invent anything. If something is unclear, say so.

Return ONLY a JSON object with these keys:
  "reason_for_call": one short sentence
  "outcome": one short sentence on how the call ended
  "actions_taken": list of short strings (bookings, cancellations, requests, escalations)
  "follow_up_needed": true or false (true if staff need to do something)
  "follow_up_details": what staff need to do, or an empty string
  "urgency": one of "none", "routine", "urgent", "emergency"
  "red_flags_mentioned": true if the patient described any emergency warning sign
  "patient_verified": true or false
"""


def transcript_from_context(context) -> list[str]:
    """Turn the conversation into 'Patient: ...' and 'Lara: ...' lines."""
    try:
        messages = context.get_messages()
    except Exception:  # noqa: BLE001
        messages = getattr(context, "messages", []) or []
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        speaker = {"user": "Patient", "assistant": "Lara"}.get(message.get("role"))
        if speaker is None:
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if isinstance(content, str) and content.strip():
            lines.append(f"{speaker}: {content.strip()}")
    return lines


def _events_text(state) -> str:
    return "\n".join(
        f"- {e['tool']}: {e['outcome']}"
        + (f" ({e['spoken_when']})" if e.get("spoken_when") else "")
        + (f" [{e['reason']}]" if e.get("reason") else "")
        for e in state.events
        if e["tool"] not in ("search_knowledge_base", "search_health_info", "get_surgery_status")
    ) or "- none"


def fallback_summary(state, transcript: list[str]) -> dict:
    """A plain, factual record built only from what the tools did."""
    urgency, actions, follow_up = "none", [], False
    for event in state.events:
        if event["tool"] == "escalate_to_human":
            urgency = event["outcome"] if event["outcome"] in ("urgent", "emergency") else (
                urgency if urgency != "none" else "routine")
            actions.append(f"Escalated to staff ({event['outcome']}): {event.get('reason', '')}")
            follow_up = True
        elif event["tool"] == "request_callback":
            actions.append("Callback requested")
            follow_up = True
        elif event["outcome"] in ("booked", "cancelled", "moved", "requested"):
            actions.append(f"{event['tool']}: {event['outcome']} {event.get('spoken_when', '')}".strip())
    return {
        "reason_for_call": "Not captured (automatic summary was unavailable).",
        "outcome": f"Call had {len(transcript)} lines of conversation.",
        "actions_taken": actions,
        "follow_up_needed": follow_up,
        "follow_up_details": "See the open requests list." if follow_up else "",
        "urgency": urgency,
        "red_flags_mentioned": urgency == "emergency",
        "patient_verified": state.patient is not None,
        "source": "fallback",
    }


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _tidy(data: dict, state) -> dict:
    """Make sure the model's JSON has the right shape, so the staff page never breaks."""
    urgency = data.get("urgency")
    return {
        "reason_for_call": str(data.get("reason_for_call") or "Not stated"),
        "outcome": str(data.get("outcome") or "Not stated"),
        "actions_taken": [str(a) for a in (data.get("actions_taken") or []) if a][:10],
        "follow_up_needed": bool(data.get("follow_up_needed")),
        "follow_up_details": str(data.get("follow_up_details") or ""),
        "urgency": urgency if urgency in ("none", "routine", "urgent", "emergency") else "none",
        "red_flags_mentioned": bool(data.get("red_flags_mentioned")),
        # What actually happened beats what the model believes happened.
        "patient_verified": state.patient is not None,
        "source": "gemini",
    }


async def _llm_summary(api_key: str, model: str, transcript: list[str], state) -> dict | None:
    from google import genai
    from google.genai import types

    prompt = (
        f"{_SUMMARY_INSTRUCTIONS}\nActions recorded by the system:\n{_events_text(state)}\n\n"
        "Transcript:\n" + "\n".join(transcript)
    )
    client = genai.Client(api_key=api_key)
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0
            ),
        ),
        timeout=SUMMARY_TIMEOUT_SECONDS,
    )
    data = _parse_json(response.text)
    return _tidy(data, state) if data else None


async def finish_call(context, state, api_key: str | None, model: str) -> None:
    """Summarise the call and save it. Safe to call at the end of every call."""
    try:
        transcript = transcript_from_context(context)
        if not any(line.startswith("Patient:") for line in transcript):
            logger.info(f"[{state.session_id}] No patient speech, so no call summary saved")
            return
        summary = None
        if api_key:
            try:
                summary = await _llm_summary(api_key, model, transcript, state)
            except Exception as error:  # noqa: BLE001
                logger.warning(f"[{state.session_id}] Gemini summary failed: {error!r}")
        if summary is None:
            summary = fallback_summary(state, transcript)
        db.save_call_summary(
            state.session_id, state.patient["id"] if state.patient else None, summary
        )
        logger.info(f"[{state.session_id}] Call summary saved ({summary['source']}): {summary['reason_for_call']}")
    except Exception:  # noqa: BLE001
        logger.exception("Could not save the call summary")
