"""Turn a Lara terminal log into a latency report.

How to use:
    1. Run the agent and keep its output:
         uv run bot.py 2>&1 | Tee-Object run.log        (PowerShell)
         uv run bot.py 2>&1 | tee run.log               (Mac / Linux)
       (You can also just copy the terminal text into a file.)
    2. Have a conversation, then stop the agent with Ctrl+C.
    3. Run:   uv run python latency_report.py run.log

For every turn it shows how long the caller waited in silence, from the moment
they stopped speaking to the moment Lara started speaking, and where that time
went. With "one moment" fillers switched on, the first sound is the filler, so
the silence shown is time to first sound, not time to the real answer.
"""

import re
import statistics
import sys

# Not anchored to the start of the line: PowerShell can put text such as "uv : " in front of a log line.
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \|")
PATTERNS = {
    "stop": re.compile(r"User stopped speaking"),
    "llm_ttfb": re.compile(r"\w*LLMService#\d+ TTFB: ([\d.]+)s"),
    "tokens": re.compile(r"\w*LLMService#\d+ prompt tokens: (\d+), completion tokens: (\d+)"),
    "tool": re.compile(r"Calling function \[(\w+):"),
    "tts_first_audio": re.compile(r"TTSService#\d+ TTFA: ([\d.]+)s"),
    "speak": re.compile(r"Bot started speaking"),
}
# Pipecat's own wording for "the user stopped talking" can appear in several places; only
# the aggregator's decision to end the turn counts.
STOP_MARKER = "_on_user_turn_stopped"


def read_text(path: str) -> str:
    """Read a log, even if PowerShell saved it as UTF-16."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:200]:
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def _seconds(stamp: str) -> float:
    date_part, time_part = stamp.split(" ")
    hours, minutes, seconds = time_part.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        match = STAMP.search(line)
        if not match:
            continue
        at = _seconds(match.group(1))
        for kind, pattern in PATTERNS.items():
            found = pattern.search(line)
            if not found:
                continue
            if kind == "stop" and STOP_MARKER not in line:
                continue
            events.append({"at": at, "kind": kind, "groups": found.groups()})
            break
    return events


def _is_cancelled(index: int, events: list[dict]) -> bool:
    """A model call that was interrupted logs zero tokens right next to its TTFB line."""
    here = events[index]["at"]
    for other in events[max(0, index - 3): index + 4]:
        if other["kind"] == "tokens" and int(other["groups"][0]) == 0 and abs(other["at"] - here) <= 0.05:
            return True
    return False


def build_turns(events: list[dict]) -> list[dict]:
    turns, last_stop, window = [], None, []
    for index, event in enumerate(events):
        if event["kind"] == "stop":
            last_stop, window = event["at"], []
        elif last_stop is not None and event["kind"] != "speak":
            window.append((index, event))
        elif event["kind"] == "speak" and last_stop is not None:
            llm = [float(e["groups"][0]) for i, e in window
                   if e["kind"] == "llm_ttfb" and not _is_cancelled(i, events)]
            tokens = [int(e["groups"][0]) for _, e in window if e["kind"] == "tokens"]
            first_audio = [float(e["groups"][0]) for _, e in window if e["kind"] == "tts_first_audio"]
            turns.append({
                "silence": event["at"] - last_stop,
                "llm_calls": llm,
                "tools": [e["groups"][0] for _, e in window if e["kind"] == "tool"],
                "prompt_tokens": max(tokens) if tokens else 0,
                "tts_first_audio": first_audio[0] if first_audio else None,
            })
            last_stop = None
    return turns


def report(turns: list[dict]) -> str:
    if not turns:
        return "No finished turns found. Does the log include the DEBUG lines Pipecat prints?"
    lines = [f"{'#':>2} {'silence':>8}  {'model calls':<14} {'tools':<24} {'tokens':>6} {'voice':>6}", "-" * 66]
    for number, turn in enumerate(turns, 1):
        calls = " + ".join(f"{x:.1f}" for x in turn["llm_calls"]) or "-"
        tools = ", ".join(turn["tools"]) or "-"
        voice = f"{turn['tts_first_audio']:.1f}s" if turn["tts_first_audio"] is not None else "-"
        lines.append(f"{number:>2} {turn['silence']:7.1f}s  {calls:<14} {tools:<24} {turn['prompt_tokens']:>6} {voice:>6}")
    silences = [t["silence"] for t in turns]
    model_time = [sum(t["llm_calls"]) for t in turns]
    lines += [
        "-" * 66,
        f"turns {len(turns)} | silence: median {statistics.median(silences):.1f}s, "
        f"mean {statistics.mean(silences):.1f}s, worst {max(silences):.1f}s",
        f"the language model is {sum(model_time) / sum(silences):.0%} of the waiting time",
    ]
    with_tools = [t["silence"] for t in turns if t["tools"]]
    without = [t["silence"] for t in turns if not t["tools"]]
    if with_tools and without:
        lines.append(f"turns with a tool: median {statistics.median(with_tools):.1f}s | without: {statistics.median(without):.1f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(report(build_turns(parse_events(read_text(sys.argv[1])))))