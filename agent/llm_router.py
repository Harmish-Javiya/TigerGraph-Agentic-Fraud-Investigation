"""
Provider chain: local Qwen3 -> Gemini Flash -> Groq Llama.
Falls through on quota errors, connection errors, or invalid output.
"""
import os
import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = [
    {
        "name": "gemini",
        "client": OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                         api_key=os.getenv("GEMINI_API_KEY", "missing"), timeout=60),
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "no_think": False,
    },
    {
        "name": "local",
        "client": OpenAI(base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
                         api_key="ollama", timeout=240),
        "model": os.getenv("LOCAL_MODEL", "qwen3:4b-instruct"),
        "no_think": False,
    },
    {
        "name": "groq",
        "client": OpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.getenv("GROQ_API_KEY", "missing"), timeout=60),
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "no_think": False,
    },
]
_cooldown_until = {p["name"]: 0.0 for p in PROVIDERS}
TOKENS = 0
LAST_PROVIDER = ""


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Parse JSON even if wrapped in ``` fences or surrounded by prose."""
    text = _strip_think(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _classify(err: Exception) -> tuple[str, float]:
    """Returns (kind, cooldown_seconds)."""
    msg = str(err).lower()
    if any(k in msg for k in ("per day", "daily", "tpd", "rpd")):
        return "quota_daily", 3600
    if any(k in msg for k in ("429", "rate_limit", "rate limit", "resource_exhausted", "quota")):
        return "quota_minute", 65
    if any(k in msg for k in ("connection", "refused", "timed out", "timeout")):
        return "down", 120
    return "other", 0


def chat(prompt: str, system: str = "", json_mode: bool = False,
         max_tokens: int = 2000) -> tuple[str, int]:
    """Returns (text, tokens_used). Raises RuntimeError only if every provider fails."""
    global TOKENS, LAST_PROVIDER
    errors = []

    for p in PROVIDERS:
        if time.time() < _cooldown_until[p["name"]]:
            continue
        if p["name"] != "local" and p["client"].api_key == "missing":
            continue

        user_content = prompt + ("\n/no_think" if p["no_think"] else "")
        messages = ([{"role": "system", "content": system}] if system else []) \
                   + [{"role": "user", "content": user_content}]

        try:
            resp = p["client"].chat.completions.create(
                model=p["model"], messages=messages,
                temperature=0.1, max_tokens=max_tokens,
            )
            text = _strip_think(resp.choices[0].message.content or "")
            if not text:
                raise ValueError("empty response")
            if json_mode:
                extract_json(text)  # validate; falls through to next provider if bad
            used = resp.usage.total_tokens if resp.usage else 0
            TOKENS += used
            LAST_PROVIDER = p["name"]
            return text, used
        except Exception as e:
            kind, cooldown = _classify(e)
            if cooldown:
                _cooldown_until[p["name"]] = time.time() + cooldown
            errors.append(f"{p['name']}[{kind}]: {str(e)[:150]}")
            print(f"  [LLM] {p['name']} failed ({kind}) -> trying next")

    raise RuntimeError("All LLM providers failed: " + " | ".join(errors))


if __name__ == "__main__":
    text, used = chat('Reply with JSON {"ok": true}', json_mode=True)
    print(LAST_PROVIDER, used, text)