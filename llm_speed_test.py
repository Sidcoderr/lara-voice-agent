"""How fast is each language model option, using Lara's real prompt?

It measures time to first token: how long until the model starts answering, which is
the part your caller waits through. It sends the real system prompt plus the real tool
descriptions (as text, for realistic size) and a short patient message, five times.

    uv run python llm_speed_test.py                  Gemini, using your .env
    uv run python llm_speed_test.py groq             needs GROQ_API_KEY (add --model NAME for another model)
    uv run python llm_speed_test.py google groq      several in one go
    uv run python llm_speed_test.py openai --model MODEL [--base-url URL] [--key-env NAME]
                                                     OpenAI or any OpenAI-compatible service
    add --runs 8 to take more samples
    add --list-models to see which model names your key can actually use (no test is run)

A turn that uses a tool needs two model calls, so a good result is a median under
about one second and a worst case under about two and a half. If the results are bad
only for one provider, the provider is the problem and not your setup.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from llm_options import reasoning_effort_for

# Do not touch the real demo database just to read the tool descriptions.
os.environ.setdefault("SURGERY_DB_PATH", os.path.join(tempfile.mkdtemp(), "speedtest.db"))

load_dotenv()

DEFAULT_GOOGLE_MODEL = "gemini-3.1-flash-lite"
USER_MESSAGE = "Hi, I'd like to book a GP appointment for tomorrow, please."
PAUSE_BETWEEN_RUNS = 1.5


@dataclass
class Spec:
    label: str
    kind: str  # "google" or "openai"
    model: str
    api_key: str | None
    base_url: str | None = None
    reasoning_effort: str | None = None


def realistic_system_text() -> str:
    """Lara's real prompt, plus her real tool descriptions as text so the size is realistic."""
    from loguru import logger

    logger.remove()
    from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper

    import prompts
    from tools import SessionState, build_tools

    schemas = []
    for tool in build_tools(SessionState()):
        schema = DirectFunctionWrapper(tool).to_function_schema()
        schemas.append({"name": schema.name, "description": schema.description,
                        "properties": schema.properties, "required": schema.required})
    return prompts.build_system_prompt() + "\n\n## Tool descriptions (included for size)\n" + json.dumps(schemas)


# ------------------------------------------------------------------ one timed request


async def time_one_google(spec: Spec, system: str) -> float:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=spec.api_key)
    started = time.perf_counter()
    stream = await client.aio.models.generate_content_stream(
        model=spec.model,
        contents=USER_MESSAGE,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=60,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    async for chunk in stream:
        if chunk.text:
            return time.perf_counter() - started
    raise RuntimeError("the model returned no text")


def _openai_client(spec: Spec):
    """An OpenAI-style client with a forgiving connection timeout (a speed test should
    report how slow something is, not give up after 5 seconds on a slow network)."""
    import httpx
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=spec.api_key, base_url=spec.base_url, max_retries=0,
                       timeout=httpx.Timeout(60.0, connect=15.0))


async def time_one_openai(spec: Spec, system: str) -> float:
    client = _openai_client(spec)
    try:
        started = time.perf_counter()
        extra = {"reasoning_effort": spec.reasoning_effort} if spec.reasoning_effort else {}
        stream = await client.chat.completions.create(
            model=spec.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": USER_MESSAGE}],
            stream=True,
            **extra,
        )
        try:
            async for chunk in stream:
                if chunk.choices and (chunk.choices[0].delta.content or chunk.choices[0].delta.tool_calls):
                    return time.perf_counter() - started
        finally:
            close = getattr(stream, "close", None)
            if close:
                await close()
        raise RuntimeError("the model returned no text")
    finally:
        await client.close()


async def measure(spec: Spec, system: str, runs: int) -> dict:
    timer = time_one_google if spec.kind == "google" else time_one_openai
    times, errors = [], []
    for attempt in range(runs):
        try:
            times.append(await timer(spec, system))
        except Exception as error:  # noqa: BLE001
            errors.append(f"{type(error).__name__}: {str(error)[:90]}")
        if attempt < runs - 1:
            await asyncio.sleep(PAUSE_BETWEEN_RUNS)
    return {"spec": spec, "times": times, "errors": errors}


# ------------------------------------------------------------------ listing models

_NOT_CHAT = ("whisper", "tts", "orpheus", "guard", "safeguard", "embed", "moderation", "playai")


async def list_models(spec: Spec) -> list[str]:
    """The model names this key can use, with speech and safety models left out."""
    if spec.kind == "google":
        from google import genai

        client = genai.Client(api_key=spec.api_key)
        names = [m.name.removeprefix("models/") async for m in await client.aio.models.list()]
    else:
        client = _openai_client(spec)
        try:
            names = [m.id for m in (await client.models.list()).data]
        finally:
            await client.close()
    return sorted(n for n in names if not any(word in n.lower() for word in _NOT_CHAT))


# ------------------------------------------------------------------ reporting


def describe(result: dict) -> str:
    spec, times, errors = result["spec"], result["times"], result["errors"]
    lines = []
    if times:
        shown = " ".join(f"{t:.1f}s" for t in times)
        lines.append(f"{spec.label:34} first token: {shown}")
        lines.append(f"{'':34} median {statistics.median(times):.1f}s | worst {max(times):.1f}s"
                     + (f" | {len(errors)} failed" if errors else ""))
    else:
        lines.append(f"{spec.label:34} every request failed")
    for message in sorted(set(errors)):
        lines.append(f"{'':34} error: {message}")
    return "\n".join(lines)


def build_specs(args) -> list[Spec]:
    specs = []
    for name in args.providers or ["google"]:
        if name == "google":
            model = os.getenv("GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL)
            specs.append(Spec(f"google:{model}", "google", model, os.getenv("GOOGLE_API_KEY")))
        elif name == "groq":
            model = args.model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            effort = reasoning_effort_for(model, args.reasoning_effort or os.getenv("GROQ_REASONING_EFFORT"))
            label = f"groq:{model}" + (f" (reasoning: {effort})" if effort else "")
            specs.append(Spec(label, "openai", model, os.getenv("GROQ_API_KEY"),
                              "https://api.groq.com/openai/v1", effort))
        elif name == "openai":
            model = args.model or os.getenv("OPENAI_MODEL", "")
            if not model:
                sys.exit("openai needs --model (or OPENAI_MODEL): the model name from your provider's list.")
            base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
            key = os.getenv(args.key_env or "OPENAI_API_KEY")
            effort = reasoning_effort_for("", args.reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT"))
            specs.append(Spec(f"openai:{model}" + (f" at {base_url}" if base_url else ""), "openai", model, key, base_url, effort))
        else:
            sys.exit(f"Unknown provider '{name}'. Use google, groq or openai.")
    return specs


async def main(args) -> None:
    system = realistic_system_text()
    if not args.list_models:
        print(f"Request size: about {len(system) // 4:,} tokens (real calls grow as the conversation does).\n")
    for spec in build_specs(args):
        if not spec.api_key:
            print(f"{spec.label:34} skipped: no API key found in your .env\n")
            continue
        if args.list_models:
            try:
                names = await list_models(spec)
                print(f"Models your {spec.label.split(':')[0]} key can use (speech and safety models hidden):")
                print("\n".join(f"  {name}" for name in names) + "\n")
            except Exception as error:  # noqa: BLE001
                print(f"Could not list models: {type(error).__name__}: {str(error)[:120]}\n")
            continue
        print(describe(await measure(spec, system, args.runs)) + "\n")
    if not args.list_models:
        print("Aim for a median under about 1s and a worst case under about 2.5s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("providers", nargs="*", help="google, groq and/or openai")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--list-models", action="store_true", help="show the models your key can use, then stop")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--key-env")
    parser.add_argument("--reasoning-effort", help="override how much a reasoning model thinks (low, none, medium...); 'omit' sends nothing")
    asyncio.run(main(parser.parse_args()))