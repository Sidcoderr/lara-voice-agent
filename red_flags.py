"""A safety net for emergencies that does not wait for the language model.

Why this exists: in a real test the caller said "I've got chest pain" in short
bursts, which kept cancelling the model's replies, and the "call 999" advice
came 34 seconds later. A language model is the wrong thing to rely on alone for
the one sentence that must never be late.

So this sits between speech recognition and the model. When what the caller
says contains an emergency phrase, it immediately:
  1. speaks a short, fixed message (999, or Samaritans for self-harm),
  2. records an emergency alert for the staff desk,
  3. tells the model what already happened, so it does not repeat itself.

It is deliberately simple and cautious. It will sometimes fire when someone
only mentions a symptom (the messages are worded "if you have ... right now"
so a false alarm does no harm), and it will miss unusual wording, which is why
the model keeps its own emergency rules as well. It fires once per kind of
emergency per call. Turn it off with RED_FLAGS=off.
"""

import re
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import surgery_db as db

_NO_NOSE = r"(?!\s+(?:through|out of|via)\s+(?:my\s+|the\s+)?nose)"  # a blocked nose is not an emergency

CALL_999 = "please call 999 straight away"

# (category, message spoken to the caller, patterns, what to tell staff the caller was told)
RED_FLAGS = [
    ("chest pain",
     f"If you have chest pain, {CALL_999}.",
     [r"\bchest pains?\b", r"\bpains? in (?:my|the) chest\b",
      r"\b(?:tight(?:ness)?|pressure|heaviness) (?:in|on) (?:my|the) chest\b",
      r"\bmy chest (?:hurts|is hurting|is tight|feels tight|feels heavy)\b",
      r"\bheart attack\b"]),
    ("trouble breathing",
     f"If you're struggling to breathe, {CALL_999}.",
     [rf"\b(?:can'?t|cannot|can not|unable to) breathe\b{_NO_NOSE}",
      r"\bcan'?t catch my breath\b",
      rf"\b(?:struggling|difficulty|trouble|hard) (?:to |with )?breath(?:e|ing)\b{_NO_NOSE}",
      r"\b(?:not|stopped) breathing\b", r"\bshort of breath\b", r"\bgasping\b"]),
    ("possible stroke",
     f"Those could be signs of a stroke. Please {CALL_999.replace('please ', '')}.",
     [r"\b(?:having|had) a stroke\b", r"\bface (?:is )?(?:drooping|dropping)\b",
      r"\b(?:drooping|dropping) face\b", r"\bslurred speech\b", r"\bslurring (?:my )?words\b",
      r"\b(?:can'?t|cannot) (?:lift|move) (?:my|his|her) (?:arm|leg)\b"]),
    ("collapse",
     f"If someone has collapsed or isn't responding, {CALL_999}.",
     [r"\bunconscious\b", r"\bpassed out\b", r"\bcollapsed\b", r"\bblacked out\b",
      r"\bnot (?:responding|waking up)\b", r"\bwon'?t wake up\b"]),
    ("seizure",
     f"If someone is having a seizure, {CALL_999}.",
     [r"\bseizures?\b", r"\bhaving a fit\b", r"\bconvulsing\b", r"\bfitting\b"]),
    ("heavy bleeding",
     f"If the bleeding is heavy or won't stop, {CALL_999}.",
     [r"\b(?:bleeding|blood) (?:heavily|badly|a lot|everywhere)\b", r"\bheavy bleeding\b",
      r"\bsevere bleeding\b", r"\bwon'?t stop bleeding\b", r"\bbleeding won'?t stop\b",
      r"\blosing (?:a lot of )?blood\b", r"\b(?:vomiting|throwing up) blood\b"]),
    ("choking",
     f"If someone is choking, {CALL_999}.",
     [r"\bchoking\b"]),
    ("overdose",
     f"If anyone has taken too much of a medicine or swallowed something harmful, {CALL_999}.",
     [r"\boverdos(?:e|ed|ing)\b", r"\btook too many (?:tablets|pills)\b",
      r"\bswallowed (?:poison|bleach|pills|tablets)\b"]),
    ("self-harm",
     "I'm really sorry you're feeling like this. If you're in immediate danger, please call 999. "
     "You can also call Samaritans, free, at any time, on 116 123.",
     [r"\bkill(?:ing)? myself\b", r"\bsuicid(?:e|al)\b", r"\bend(?:ing)? my (?:own )?life\b",
      r"\bwant(?:ed)? to die\b", r"\b(?:hurt|harm|hurting|harming) myself\b",
      r"\bdon'?t want to (?:live|be here|be alive)\b", r"\bno reason to live\b"]),
]

_COMPILED = [(name, message, [re.compile(p) for p in patterns]) for name, message, patterns in RED_FLAGS]

_NEGATORS = {
    "no", "not", "never", "without", "nothing", "don't", "dont", "didn't", "didnt", "doesn't",
    "doesnt", "haven't", "havent", "hasn't", "hasnt", "isn't", "isnt", "wasn't", "wasnt",
    "aren't", "arent", "weren't", "werent", "won't", "wont",
}


@dataclass(frozen=True)
class Alert:
    category: str
    matched: str
    message: str


def _normalise(text: str) -> str:
    text = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip()


def _negated(text_before: str) -> bool:
    """Is the phrase being denied ("I don't have chest pain")? Looks within the same clause only."""
    clause = re.split(r"[.,;:!?]", text_before)[-1]
    return any(word in _NEGATORS for word in re.findall(r"[a-z']+", clause)[-3:])


def detect(text: str) -> Alert | None:
    """The first emergency phrase in what the caller said, or None."""
    lowered = _normalise(text)
    for category, message, patterns in _COMPILED:
        for pattern in patterns:
            for match in pattern.finditer(lowered):
                if not _negated(lowered[: match.start()]):
                    return Alert(category, match.group(0), message)
    return None


def record_emergency(state, alert: Alert) -> int:
    """Save an emergency alert for the staff desk (once per call; later alerts are added to it)."""
    told = "Samaritans and 999" if alert.category == "self-harm" else "999"
    reason = (f"Automatic safety alert ({alert.category}): the caller said \"{alert.matched}\". "
              f"The caller was told to contact {told}.")
    patient = state.patient
    if state.emergency_id is not None:
        db.attach_callback_details([state.emergency_id], extra_reason=reason)
        return state.emergency_id
    callback_id = db.add_callback(
        "escalation", "emergency", reason,
        patient_id=patient["id"] if patient else None,
        caller_name=patient["full_name"] if patient else "",
        phone_number=patient["phone"] if patient else "",
    )
    state.emergency_id = callback_id
    state.escalation_ids.append(callback_id)
    state.log("safety_net", "emergency", category=alert.category, callback_id=callback_id)
    return callback_id


def model_note(alert: Alert) -> str:
    """What to tell the language model, so it does not repeat or contradict the safety message."""
    if alert.category == "self-harm":
        return (
            "SAFETY NOTICE (automatic): the caller's words suggest they may be thinking of harming "
            "themselves. The system has already spoken a kind message giving 999 and Samaritans (116 123) "
            "and recorded an emergency alert. Do not escalate again or ask for personal details. Reply "
            "calmly and kindly, ask if they are safe right now, and say again that 999 or Samaritans can help."
        )
    return (
        f"SAFETY NOTICE (automatic): the caller's words matched an emergency warning ({alert.category}). "
        "The system has already told them to call 999 and recorded an emergency alert, so do not call "
        "escalate_to_human again for this and do not ask for personal details. Reply calmly and briefly: "
        "check they are getting help, and tell them again to call 999 if they have not."
    )


class RedFlagProcessor(FrameProcessor):
    """Sits between speech recognition and the model, and acts the moment an emergency phrase is heard."""

    def __init__(self, state, context=None, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            alert = detect(frame.text)
            if alert and alert.category not in self._state.alerted:
                self._state.alerted.add(alert.category)
                try:
                    record_emergency(self._state, alert)
                    if self._context is not None:
                        self._context.add_message({"role": "developer", "content": model_note(alert)})
                except Exception:  # noqa: BLE001
                    logger.exception("Safety net could not save the alert; still speaking the message")
                logger.warning(f"[{self._state.session_id}] SAFETY NET: {alert.category} ({alert.matched!r})")
                speak = TTSSpeakFrame(alert.message)
                speak.interruptible = False  # must not be lost if the caller keeps talking
                await self.push_frame(speak)
        await self.push_frame(frame, direction)