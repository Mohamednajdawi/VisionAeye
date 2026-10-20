"""
L3 -- on-demand deliberate assistant for the VisionAeye pipeline.

Unlike L1/L2 (TypeSafe Decisions API, calibrated structured questions:
noul/score/choice/box -- NOT designed for freeform text, see
https://docs.typesafe.ai/introduction/machine-learning-primer.md), L3
needs open-ended natural-language answers ("what's on this menu",
"where's the oat milk, read the shelf labels"). That's a different
model family: a vision-capable chat completion model via OpenRouter's
standard chat API, not the Decisions API.

Explicitly user-triggered, never continuous -- see ROADMAP.md Epic 5.
Running this on every frame would be slow and expensive; it only runs
when the user asks a question.
"""
import json
import os
import time
import urllib.error
import urllib.request

CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"  # vision-capable, cheap; swappable per call

SYSTEM_PROMPT = (
    "You are a visual assistant for a blind or low-vision person wearing "
    "a camera. They will ask you a question about what the camera sees -- "
    "reading a menu, finding a product, describing a label, reading a "
    "sign. Answer concisely and practically, in spoken-language style (no "
    "markdown, no bullet points -- this will be read aloud by "
    "text-to-speech). If you cannot find what they're asking about in the "
    "image, say so plainly rather than guessing. Give spatial directions "
    "relative to the image (left, right, top, bottom, center) when "
    "relevant, e.g. 'the oat milk is on the left side, second shelf from "
    "the bottom.'"
)


class L3Error(RuntimeError):
    """Raised on a transport/API failure. Callers must treat this as
    'could not answer' and relay that to the user via TTS, never as
    silence (a silent failure on a user-initiated question is worse
    than an honest 'I couldn't process that')."""


def ask_about_image(
    image_b64: str,
    question: str,
    api_key: str = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> dict:
    """One L3 call: a user question about the current camera frame.
    Returns {"answer": str, "lat": float, "cost": float|None, "model": str}.
    Raises L3Error on transport failure -- callers must handle this and
    speak an honest failure message, not go silent.
    """
    api_key = api_key or os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        CHAT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise L3Error(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise L3Error(f"Network error: {e}") from e
    lat = round(time.time() - t0, 3)

    try:
        answer = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise L3Error(f"Malformed chat response: {body}") from e

    usage = body.get("usage", {})
    return {
        "answer": answer,
        "lat": lat,
        "cost": usage.get("total_cost"),  # OpenRouter-specific field, may be absent
        "model": body.get("model", model),
    }
