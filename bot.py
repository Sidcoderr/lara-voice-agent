"""
Patient Support & Scheduling voice agent.

Pipeline: browser mic -> Deepgram (STT) -> Gemini (LLM, with tools) ->
ElevenLabs (TTS) -> browser speaker.

Run locally with:
    uv run bot.py
Then open http://localhost:7860/client to talk to Lara, and
http://localhost:7860/staff to see what she has handled.
"""

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.workers.runner import WorkerRunner
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.processors.frameworks.rtvi import RTVIFunctionCallReportLevel, RTVIObserverParams
from pipecat.transcriptions.language import Language
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.transports.base_transport import TransportParams

import surgery_clock
from call_summary import finish_call
from fillers import attach_fillers
from llm_options import reasoning_effort_for
from prompts import NOT_HEARD_PROMPT, build_system_prompt, greeting_instruction, greeting_text
from red_flags import RedFlagProcessor
from stall import ErrorApology, StallWatcher, seconds_from_env, stall_seconds_from_env
from tools import SessionState, build_tools

load_dotenv(override=True)

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def build_llm(system_prompt: str):
    """Pick the language model with LLM_PROVIDER (default: google).

    google : Gemini. Needs GOOGLE_API_KEY (and optionally GOOGLE_MODEL).
    groq   : Groq's hosted open models. Needs GROQ_API_KEY (and optionally GROQ_MODEL,
             GROQ_REASONING_EFFORT). Model names change often: check console.groq.com/docs/models.
    openai : OpenAI, or ANY OpenAI-compatible service (Cerebras, Together, Fireworks...).
             Needs OPENAI_API_KEY and OPENAI_MODEL, plus OPENAI_BASE_URL for anything
             that is not OpenAI itself.

    The rest of the pipeline is identical whichever you choose.
    """
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()
    if provider == "groq":
        from pipecat.services.groq.llm import GroqLLMService

        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        effort = reasoning_effort_for(model, os.getenv("GROQ_REASONING_EFFORT"))
        llm = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            settings=GroqLLMService.Settings(
                model=model,
                system_instruction=system_prompt,
                extra={"reasoning_effort": effort} if effort else {},
            ),
        )
        llm.supports_developer_role = False  # send the greeting instruction as a user message
        return llm, f"groq:{model}" + (f" (reasoning: {effort})" if effort else "")
    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        model = os.getenv("OPENAI_MODEL", "").strip()
        if not model:
            raise ValueError(
                "LLM_PROVIDER=openai needs OPENAI_MODEL (the model name from your provider's list)."
            )
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        effort = reasoning_effort_for("", os.getenv("OPENAI_REASONING_EFFORT"))
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            settings=OpenAILLMService.Settings(
                model=model,
                system_instruction=system_prompt,
                extra={"reasoning_effort": effort} if effort else {},
            ),
        )
        if base_url and "api.openai.com" not in base_url:
            llm.supports_developer_role = False  # most other services do not accept that role
        return llm, f"openai:{model}" + (f" at {base_url}" if base_url else "")
    model = os.getenv("GOOGLE_MODEL", DEFAULT_MODEL)
    # If the first word has not arrived after LLM_RETRY_SECS (default 6), send the request once more: a
    # slow request is often a stuck one. LLM_RETRY_SECS=off turns this off.
    retry_secs = seconds_from_env(os.getenv("LLM_RETRY_SECS"), 6.0)
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(model=model, system_instruction=system_prompt),
        retry_on_timeout=retry_secs > 0,
        retry_timeout_secs=retry_secs if retry_secs > 0 else 5.0,
    )
    return llm, f"google:{model}"


def build_vad():
    """Voice activity detection: hears the start and end of speech in the audio itself.

    Without it, the bot only learns that you have started or stopped speaking when
    the speech-to-text service sends a transcript, about a second late, so it is
    slow to stop talking when interrupted and slow to notice you have finished.
    If it cannot be loaded the bot carries on without it. VAD=off turns it off.
    """
    if os.getenv("VAD", "on").strip().lower() == "off":
        return None
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        return SileroVADAnalyzer()
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Voice activity detection is unavailable ({error!r}); carrying on without it.")
        return None


def build_empty_turn_config() -> EmptyUserTurnConfig:
    """What Lara does when the caller clearly spoke but no words were recognised.

    By default Pipecat only answers such a turn if it interrupted Lara, and stays silent otherwise, which
    feels like being ignored. Here, when she was waiting for the caller, she says she did not catch it
    and asks them to repeat. At most two such replies in a row (a cough or noise should not start a loop).
    NOT_HEARD=off goes back to Pipecat's default.
    """
    if os.getenv("NOT_HEARD", "on").strip().lower() == "off":
        return EmptyUserTurnConfig()
    return EmptyUserTurnConfig(idle_prompt=NOT_HEARD_PROMPT, max_consecutive_recoveries=2)


def build_rtvi_params() -> RTVIObserverParams | None:
    """Keep function calls out of the browser's event stream, so the call screen stays clean.

    (By default Pipecat tells the browser every time a tool is called.) SHOW_TOOL_EVENTS=on shows them again.
    """
    if os.getenv("SHOW_TOOL_EVENTS", "off").strip().lower() in ("on", "yes", "true", "1"):
        return None
    return RTVIObserverParams(function_call_report_level={"*": RTVIFunctionCallReportLevel.DISABLED})


def build_stt():
    """Deepgram speech recognition, with two optional extras that are off unless you set them.

    DEEPGRAM_LANGUAGE   a language code such as en-IN (Indian English); Nova-3 supports it.
    DEEPGRAM_KEYTERMS   comma-separated names and words to listen for, such as
                        "Lara,Riverside Surgery,Sid". Nova-3 only; helps with names.
    With neither set, this is exactly the plain Deepgram setup used until now.
    """
    options = {}
    code = os.getenv("DEEPGRAM_LANGUAGE", "").strip()
    if code:
        try:
            options["language"] = Language(code)
        except ValueError:
            logger.warning(f"DEEPGRAM_LANGUAGE={code!r} is not a language code I know; using the default.")
    terms = [term.strip() for term in os.getenv("DEEPGRAM_KEYTERMS", "").split(",") if term.strip()]
    if terms:
        options["keyterm"] = terms
    if options:
        return DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), settings=DeepgramSTTService.Settings(**options))
    return DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))


def build_tts(http_session):
    """ElevenLabs. (Pipecat already defaults to ElevenLabs' low-latency Flash model.)"""
    return ElevenLabsHttpTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        aiohttp_session=http_session,
        settings=ElevenLabsHttpTTSService.Settings(voice=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")),
    )


async def start_greeting(worker, context, surgery_open: bool) -> None:
    """Greet the caller as soon as the call is ready."""
    if os.getenv("GREETING", "fixed").strip().lower() == "model":
        # The old way: ask the language model to write the greeting (slower).
        context.add_message({"role": "developer", "content": greeting_instruction(surgery_open)})
        await worker.queue_frames([LLMRunFrame()])
        return
    # Speak fixed text straight away: no model call, so no wait. The model is told it has
    # already greeted the caller, so it does not greet them again.
    context.add_message(
        {
            "role": "developer",
            "content": "The call has just connected and you have already greeted the caller out loud. Wait for them to speak.",
        }
    )
    await worker.queue_frames([TTSSpeakFrame(greeting_text(surgery_open))])


async def run_bot(transport, runner_args: RunnerArguments):
    # --- AI services ---------------------------------------------------
    stt = build_stt()

    http_session = aiohttp.ClientSession()

    tts = build_tts(http_session)
    llm, llm_name = build_llm(build_system_prompt())
    if os.getenv("FILLERS", "on").strip().lower() != "off":
        attach_fillers(llm, tts)

    # --- Tools and conversation context -----------------------------------
    # Each call gets its own tools and its own record of who was verified.
    # Tools listed in the context are registered with the LLM automatically.
    state = SessionState()
    logger.info(f"[{state.session_id}] Using {llm_name} (prompt: {os.getenv('LARA_PROMPT', 'compact')})")
    context = LLMContext(tools=build_tools(state))
    vad = build_vad()
    logger.info(f"[{state.session_id}] Voice activity detection: {'on' if vad else 'off'}")
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad, empty_user_turn=build_empty_turn_config())
    )

    # --- Pipeline ----------------------------------------------------------
    # The safety net sits right after speech recognition, so an emergency phrase is acted on
    # at once instead of waiting for the turn detector and the language model. RED_FLAGS=off disables it.
    safety_net = []
    if os.getenv("RED_FLAGS", "on").strip().lower() != "off":
        safety_net = [RedFlagProcessor(state, context)]
    logger.info(f"[{state.session_id}] Emergency safety net: {'on' if safety_net else 'off'}")

    # If the model has not started answering a few seconds after the caller finishes, say a short
    # holding phrase so there is no dead air. STALL_SECONDS=off turns it off.
    slow_reply = []
    stall_seconds = stall_seconds_from_env(os.getenv("STALL_SECONDS"))
    if stall_seconds > 0:
        watcher = StallWatcher(stall_seconds)
        slow_reply = [watcher]

        @user_aggregator.event_handler("on_user_turn_inference_triggered")
        async def start_stall_timer(*args):
            watcher.guard.start()

    logger.info(f"[{state.session_id}] Slow-reply phrase: {f'after {stall_seconds:g}s' if slow_reply else 'off'}")

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            *safety_net,
            user_aggregator,
            llm,
            *slow_reply,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        rtvi_observer_params=build_rtvi_params(),
    )
    logger.info(f"[{state.session_id}] Not-heard reply: {'off' if os.getenv('NOT_HEARD', 'on').strip().lower() == 'off' else 'on'}"
                f" | tool events in the browser: {'shown' if build_rtvi_params() is None else 'hidden'}")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    # If the model fails outright (a 503 "high demand" or no answer at all), say a short apology
    # instead of leaving silence. ERROR_APOLOGY=off turns it off.
    apology_on = os.getenv("ERROR_APOLOGY", "on").strip().lower() != "off"
    if apology_on:
        apologise = ErrorApology(llm)

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            await apologise(worker, frame)

    logger.info(f"[{state.session_id}] Error apology: {'on' if apology_on else 'off'}")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await start_greeting(worker, context, surgery_clock.is_open())

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"[{state.session_id}] Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[{state.session_id}] Client disconnected")
        await runner.cancel()

    try:
        await runner.run()
    finally:
        await http_session.close()
        # Write the end-of-call summary for the staff desk. Never raises.
        await finish_call(context, state, os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_MODEL", DEFAULT_MODEL))


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import app, main

    # Add the reception desk page to the same server. If this ever fails,
    # Lara still works; you just lose the /staff page.
    try:
        from staff_desk import register_staff_routes

        register_staff_routes(app)
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Staff desk page not available: {error!r}")

    # A page where a patient can see their own bookings and requests ("My requests").
    # PATIENT_PAGE=off turns it off. Like the staff page, if it fails Lara still works.
    if os.getenv("PATIENT_PAGE", "on").strip().lower() != "off":
        try:
            from patient_view import register_patient_routes

            register_patient_routes(app)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Patient page not available: {error!r}")

    main()