"""Small shared choices about language models, used by the bot and by the speed test."""


def reasoning_effort_for(model: str, override: str | None = None) -> str | None:
    """How much a reasoning model should 'think' before it starts answering.

    On a phone call, thinking is dead air, so for voice we turn it down as far as
    each model family allows. GPT-OSS models cannot switch thinking off, only turn
    it down ("low"). Qwen3 models accept "none". Other models are left alone.

    Pass an override to choose yourself. "omit" (or "off") sends nothing at all,
    which is the way out if a provider rejects the setting.
    """
    if override and override.strip():
        value = override.strip().lower()
        return None if value in ("omit", "off") else value
    name = (model or "").lower()
    if name.startswith("openai/gpt-oss"):
        return "low"
    if name.startswith("qwen/"):
        return "none"
    return None
