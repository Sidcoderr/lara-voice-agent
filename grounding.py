"""Check what the model passes to a tool against what the patient actually said.

Why this exists: in testing, the model tried to verify a patient using a date
of birth it had made up, before the patient had given one. That is both
annoying (the fake attempts counted against the patient's three tries) and a
security hole (a model that guesses can stumble onto someone else's record).

So verify_patient does not take the model's word for it. It reads the patient's
own words from the conversation and only accepts a date of birth that they
really said, in digits ("21/07/1990", "21 July 1990") or in words ("the twenty
first of July nineteen ninety"). Speech recognition writes dates both ways.
"""

import datetime as dt
import re

try:  # installed with Pipecat; if it is ever missing we fall back to digits only
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_SHORT = {"sept": 9}


def _normalise(text: str) -> str:
    """Lower-case, and make 'twenty-first of July, 1990' comparable with plain words."""
    text = (text or "").lower().replace("-", " ")
    text = re.sub(r"[,.;:!?]", " ", text)
    text = re.sub(r"\band\b", " ", text)  # 'two thousand and one' == 'two thousand one'
    return re.sub(r"\s+", " ", text).strip()


def _words(number: int, style: str) -> str | None:
    if num2words is None:
        return None
    try:
        return _normalise(num2words(number, to=style))
    except Exception:  # noqa: BLE001
        return None


def _contains(text: str, form: str) -> bool:
    return bool(re.search(rf"(?<![\w]){re.escape(form)}(?![\w])", text))


def _year_forms(year: int) -> set[str]:
    forms = {str(year)}
    spoken = _words(year, "year")
    if spoken:
        forms.add(spoken)
    if 2000 <= year <= 2009 and year > 2000:  # 'twenty oh one'
        digit = _words(year - 2000, "cardinal")
        if digit:
            forms.add(f"twenty oh {digit}")
    return forms


def _day_forms(day: int) -> set[str]:
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    forms = {str(day), f"{day:02d}", f"{day}{suffix}"}
    for style in ("cardinal", "ordinal"):
        spoken = _words(day, style)
        if spoken:
            forms.add(spoken)
    return forms


def _month_forms(month: int) -> set[str]:
    name = MONTHS[month - 1]
    forms = {name, name[:3]}
    forms.update(short for short, number in _MONTH_SHORT.items() if number == month)
    return forms


def dob_was_said(dob: dt.date, patient_words: str) -> bool:
    """True if the patient's own words contain this full date of birth.

    The day, the month and the year must all be there. A date written with
    separators ('21/07/1990') counts as all three.
    """
    text = _normalise(patient_words)
    if not text:
        return False
    numeric = re.search(r"(?<!\d)(\d{1,2}) ?[/\-. ] ?(\d{1,2}) ?[/\-. ] ?(\d{2,4})(?!\d)",
                        (patient_words or "").lower())
    if numeric:
        first, second, year = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
        year = year + (1900 if year > 30 else 2000) if year < 100 else year
        if year == dob.year and {first, second} == {dob.day, dob.month}:
            return True
    year_ok = any(_contains(text, f) for f in _year_forms(dob.year))
    month_ok = any(_contains(text, f) for f in _month_forms(dob.month))
    day_ok = any(_contains(text, f) for f in _day_forms(dob.day))
    return year_ok and month_ok and day_ok


def recent_patient_words(context, last_n: int = 12) -> str | None:
    """What the patient has said recently, or None if we cannot read the conversation.

    None means "cannot check", and the caller lets the call through rather than
    blocking everyone because of a plumbing problem.
    """
    try:
        messages = context.get_messages()
    except Exception:  # noqa: BLE001
        return None
    said = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and content.strip():
            said.append(content)
    return " ".join(said[-last_n:]) if said else None


# Words a finished sentence almost never ends on. If the patient's last message ends with
# one, the sentence was probably cut off mid-thought by the turn detector.
_DANGLING_WORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "on", "for", "with", "about",
    "my", "your", "that", "this", "if", "because", "is", "are",
}


def last_patient_message(context) -> str | None:
    """The patient's most recent message, or None if the conversation cannot be read."""
    try:
        messages = context.get_messages()
    except Exception:  # noqa: BLE001
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and content.strip():
            return content
    return None


def looks_cut_off(text: str) -> bool:
    """True if the message ends on a word that cannot end a sentence ('update on the')."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    return bool(words) and words[-1] in _DANGLING_WORDS