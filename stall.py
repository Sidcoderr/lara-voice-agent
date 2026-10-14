"""A short holding phrase for when the language model is slow.

Gemini's free tier is usually quick (about 1.5 to 2 seconds) but sometimes takes
7 to 15 seconds. Silence for that long sounds like a crash. So if the model has
not started answering within a few seconds of the caller finishing, Lara says a
short phrase such as "Sorry, just a moment" instead of leaving dead air.

It only fires when nothing has happened for a while, and it cancels itself the
moment the model starts answering, the caller speaks again, or a tool starts.
It does not make the real answer any faster. STALL_SECONDS=off turns it off, or
STALL_SECONDS=8 changes how long it waits (default 5).
"""

import asyncio
import random
import time

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

PHRASES = ["Sorry, just a moment.", "Bear with me, one second."]
DEFAULT_SECONDS = 5.0
APOLOGY = "Sorry, I'm having a little trouble at the moment. Could you say that again?"


def seconds_from_env(value: str | None, default: float) -> float:
    """A number of seconds from a setting. 0 means off ('off', 'no', 'false' and '0' all switch it off)."""
    text = (value or "").strip().lower()
    if not text:
        return default
    if text in ("off", "no", "false", "0"):
        return 0.0
    try:
        return max(0.0, float(text))
    except ValueError:
        return default


def stall_seconds_from_env(value: str | None) -> float:
    """How long to wait before saying the holding phrase. 0 means off."""
    return seconds_from_env(value, DEFAULT_SECONDS)


class StallGuard:
    """A cancellable timer: start() when we begin waiting, stop() when the wait is over."""

    def __init__(self, seconds: float, on_stall):
        self.seconds = seconds
        self._on_stall = on_stall
        self._task: asyncio.Task | None = None

    @property
    def waiting(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        self.stop()
        if self.seconds > 0:
            self._task = asyncio.get_running_loop().create_task(self._wait())

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _wait(self) -> None:
        try:
            await asyncio.sleep(self.seconds)
        except asyncio.CancelledError:
            return
        self._task = None
        try:
            await self._on_stall()
        except Exception:  # noqa: BLE001
            logger.exception("Could not speak the holding phrase")


class StallWatcher(FrameProcessor):
    """Sits between the language model and speech output, and watches for a slow answer.

    The timer is started from outside when a caller's turn is handed to the model
    (the first round), and here when a tool result is handed back to it (the second
    round). It is stopped as soon as the model produces anything.
    """

    def __init__(self, seconds: float = DEFAULT_SECONDS, **kwargs):
        super().__init__(**kwargs)
        self.guard = StallGuard(seconds, self._say)

    async def _say(self) -> None:
        phrase = random.choice(PHRASES)
        logger.warning(f"Model is slow (over {self.guard.seconds:g}s): saying {phrase!r}")
        await self.push_frame(TTSSpeakFrame(phrase))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.UPSTREAM:
            self.guard.start()  # a tool result is going back to the model
        elif isinstance(
            frame,
            # Real progress only. NOT LLMFullResponseStartFrame: Pipecat sends that the moment a request
            # BEGINS, before the model has answered, so stopping on it would cancel the timer every time.
            (LLMTextFrame, FunctionCallsStartedFrame, TTSSpeakFrame, InterruptionFrame,
             UserStartedSpeakingFrame, EndFrame, CancelFrame),
        ):
            self.guard.stop()
        await self.push_frame(frame, direction)


class ErrorApology:
    """Say a short apology when the language model fails, instead of leaving the caller in silence.

    Gemini's free tier sometimes answers "503: high demand" or never answers (Pipecat gives up after 20
    seconds). Both arrive as an error from the model's own processor. This speaks once, and not again for
    `min_gap` seconds, so a run of failures does not become a run of apologies. ERROR_APOLOGY=off disables it.
    """

    def __init__(self, llm, min_gap: float = 15.0, clock=time.monotonic):
        self._llm = llm
        self._min_gap = min_gap
        self._clock = clock
        self._last: float | None = None

    async def __call__(self, worker, frame) -> bool:
        """Handler for the pipeline's on_pipeline_error event. Returns True if it spoke."""
        if not isinstance(frame, ErrorFrame) or getattr(frame, "processor", None) is not self._llm:
            return False
        now = self._clock()
        if self._last is not None and now - self._last < self._min_gap:
            return False
        self._last = now
        logger.warning(f"Model failed ({str(frame.error)[:80]}): saying a short apology")
        await worker.queue_frames([TTSSpeakFrame(APOLOGY, append_to_context=False)])
        return True