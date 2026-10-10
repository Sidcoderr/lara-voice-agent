"""Say a few words while a tool is running, so the caller is not left in silence.

A turn that needs a tool takes two trips to the language model: one to decide to
call the tool and one to word the answer. The second trip can take several
seconds, so as soon as the tool starts we play a short phrase ("One moment.")
while the model works on the real answer.

Only tools whose answer is not urgent get a filler. Nothing is said before
escalate_to_human, because an emergency should get the real message first, and
the quick lookups (search, opening hours) answer fast enough on their own.

Turn this off with FILLERS=off.
"""

import random

from pipecat.frames.frames import TTSSpeakFrame

FILLERS = {
    "verify_patient": ["Thanks, one moment while I check that.", "Thank you, just a moment."],
    "check_availability": ["Let me have a look.", "One moment, I'll check that."],
    "book_appointment": ["One moment while I book that.", "Booking that now."],
    "cancel_appointment": ["One moment while I do that."],
    "reschedule_appointment": ["One moment while I move that."],
    "request_repeat_prescription": ["One moment while I put that through."],
    "submit_admin_request": ["Got it, one moment."],
    "join_cancellation_list": ["One moment while I add you."],
}


def pick_filler(function_names: list[str], rng=random) -> str | None:
    """A short phrase for the first tool in the list that has one, else None."""
    for name in function_names:
        phrases = FILLERS.get(name)
        if phrases:
            return rng.choice(phrases)
    return None


def attach_fillers(llm, tts) -> None:
    """Make the agent speak a short phrase whenever one of these tools starts."""

    @llm.event_handler("on_function_calls_started")
    async def say_something_while_the_tool_runs(service, function_calls):
        phrase = pick_filler([call.function_name for call in function_calls])
        if phrase:
            await tts.queue_frame(TTSSpeakFrame(phrase))
