"""
PediJev -- production hazard-detection pipeline for a blind-pedestrian
smart-glasses prototype.

Architecture (deliberately split):
  prompts/<version>.json   -- the model-facing prompt: state text, model
                               id, and typed questions. Pure data, no code.
                               Swap/A-B test versions without touching this file.
  pipeline/pediev_pipeline.py (this file) -- the model-agnostic API client
                               plus the decision engine. Every threshold
                               that turns model answers into WALK/WAIT/BLOCK
                               lives here, in code, reviewable and testable
                               independently of prompt wording.

Usage:
    python pediev_pipeline.py <image.jpg> [--prompt prompts/pediev_v1_1.json]
"""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
import ssl

try:
    import certifi
    _HAS_CERTIFI = True
except Exception:
    _HAS_CERTIFI = False

DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "pediev_v1_1.json"
)


def load_prompt(prompt_path: str = DEFAULT_PROMPT_PATH) -> dict:
    """Load a versioned prompt spec: {version, model, state_text, questions}."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    for required in ("model", "state_text", "questions"):
        if required not in spec:
            raise ValueError(f"Prompt file {prompt_path} missing required field: {required}")
    return spec


def call_jev(image_b64: str, prompt_spec: dict, api_key: str, timeout: int = 60):
    """One Decisions API call using the given prompt spec.
    Returns (status, latency_s, parsed_json_or_None, raw_body)."""
    payload = {
        "model": prompt_spec["model"],
        "state": [
            {"type": "text", "text": prompt_spec["state_text"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
        "questions": prompt_spec["questions"],
    }
    req = urllib.request.Request(
        DECISIONS_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",  # required for image + box questions
        },
        method="POST",
    )
    t0 = time.time()
    # Build an SSL context using certifi when available so macOS/python
    # installations missing system CAs can still verify server certs.
    if _HAS_CERTIFI:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
    else:
        ssl_context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        # Server returned an error code with a response body
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        status = e.code
    except urllib.error.URLError as e:
        # Network/SSL-level error (e.g. certificate verification failure)
        body = str(e)
        status = None
    lat = round(time.time() - t0, 3)
    parsed = json.loads(body) if status == 200 else None
    return status, lat, parsed, body


T_STOP = 1.0


def decide(answers: dict) -> str:
    """
    PediJev v1.1 decision engine. Model perceives (via `answers`), code
    decides. Every go/no-go threshold lives here, not in the prompt --
    tunable and auditable without touching the model-facing text.

    This function is intentionally decoupled from the prompt file: it
    only depends on the QUESTION NAMES below being present in `answers`
    with the shapes the Decisions API returns (noul/score/choice/box).
    If a future prompt version renames or drops a question, this
    function needs a matching update -- that coupling is explicit, not
    hidden.

    Required question names: scene_assessable, hazard_proximity,
    hazard_type, crosswalk_present, signal_box_present,
    pedestrian_signal_color. (signal_box is captured by the prompt but
    not currently used here -- see README "known gaps".)

    Returns one of: WALK, WAIT, BLOCK, ESCALATE_UNREADABLE.
    """
    hp = answers["hazard_proximity"]["score"]
    htype = answers["hazard_type"]["choice"]
    crosswalk_present = answers["crosswalk_present"]["noul"]
    sig_present = answers["signal_box_present"]["noul"]
    sig_color = answers["pedestrian_signal_color"]["choice"]
    assessable = answers["scene_assessable"]["noul"]

    if assessable < 0.5:
        return "ESCALATE_UNREADABLE"

    # A crossing is "relevant" to this forward path if: a marked
    # crosswalk is visible, OR this is a rail/track crossing (no zebra
    # stripes but still a real crossing -- crosswalk_present is
    # unreliable there), OR hazard_proximity is already elevated
    # (evidence something ahead matters regardless of markings).
    #
    # WITHOUT this gate (v1.0 bug, found by user review): an
    # unconditional `sig_color == "red"` check let a distant, irrelevant
    # red light on another street wrongly force WAIT on someone just
    # walking down a sidewalk (crosswalk_present was correctly low, but
    # the signal check ignored it). Fixed in v1.1.
    crossing_relevant = (
        crosswalk_present >= 0.3
        or htype == "rail_or_track"
        or hp >= 0.5
    )

    if crossing_relevant and sig_color == "red":
        return "WAIT"

    # Safety exception, checked before the crosswalk rule: a vehicle
    # actively moving into/through the path has NOT yielded yet, even at
    # a crosswalk. "Crosswalk -> cars must yield" assumes the car HAS
    # yielded; a moving vehicle in frame is the one case that doesn't
    # hold.
    if htype == "moving_vehicle" and hp >= T_STOP:
        return "BLOCK"

    # Marked crosswalk + (no signal OR green) -> walk. Per Austrian StVO
    # Sec9/Sec76: at an unsignalized crosswalk, vehicles are legally
    # required to yield; a stopped/yielding vehicle there is not a
    # hazard. Excludes all non-vehicle hazard types (stairs/rail/
    # obstacle) and moving_vehicle -- a crosswalk doesn't make those
    # safe.
    if (
        crosswalk_present >= 0.5
        and (sig_color == "green" or (sig_color == "unclear" and sig_present < 0.5))
        and htype in {"none", "stopped_vehicle"}
    ):
        return "WALK"

    # Signal present but unreadable, at a relevant crossing -> never
    # fall through to WALK.
    if crossing_relevant and sig_present >= 0.5 and sig_color == "unclear":
        return "WAIT"

    if hp >= T_STOP:
        return "BLOCK"

    return "WALK"


def evaluate_image_b64(image_b64: str, prompt_spec: dict, api_key: str) -> dict:
    """Full pipeline: one API call + decision. Returns a dict with
    decision, raw answers, latency, and cost."""
    status, lat, parsed, raw_body = call_jev(image_b64, prompt_spec, api_key)
    if status != 200:
        return {"decision": "ERROR", "http_status": status, "lat": lat, "raw_body": raw_body}
    decision = decide(parsed["answers"])
    return {
        "decision": decision,
        "answers": parsed["answers"],
        "lat": lat,
        "cost": parsed["usage"]["cost"],
        "model": parsed["model"],
        "id": parsed["id"],
        "prompt_version": prompt_spec.get("version", "unknown"),
    }


def evaluate_image_file(image_path: str, prompt_path: str = DEFAULT_PROMPT_PATH, api_key: str = None) -> dict:
    """Convenience wrapper: load an image file + a prompt file, run the pipeline."""
    api_key = api_key or os.environ["OPENROUTER_API_KEY"]
    prompt_spec = load_prompt(prompt_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return evaluate_image_b64(b64, prompt_spec, api_key)


def main():
    parser = argparse.ArgumentParser(description="Run the PediJev hazard pipeline on one image.")
    parser.add_argument("image", help="Path to a JPEG frame")
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT_PATH,
        help=f"Path to a prompt JSON file (default: {DEFAULT_PROMPT_PATH})",
    )
    args = parser.parse_args()

    result = evaluate_image_file(args.image, args.prompt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
