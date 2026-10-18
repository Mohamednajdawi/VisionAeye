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
HP_CONFIDENCE_FLOOR = 0.15  # see decide() docstring + README "hazard_proximity confidence gate"
DECIDE_LOGIC_VERSION = "1.3"  # bumped independently of prompt versions -- see CHANGELOG in README


class MalformedAnswers(ValueError):
    """Raised when the model's answers dict is missing a required field
    or has the wrong shape for decide() to safely interpret. Never let
    this propagate into a WALK -- callers must treat it the same as
    ESCALATE_UNREADABLE."""


def _require(answers: dict, question: str, field: str):
    try:
        return answers[question][field]
    except (KeyError, TypeError, AttributeError) as e:
        raise MalformedAnswers(f"answers[{question!r}][{field!r}] missing or malformed") from e


def decide(answers: dict) -> str:
    """
    PediJev decision engine (logic version: see DECIDE_LOGIC_VERSION).
    Model perceives (via `answers`), code decides. Every go/no-go
    threshold lives here, not in the prompt -- tunable and auditable
    without touching the model-facing text.

    This function is intentionally decoupled from the prompt file: it
    only depends on the QUESTION NAMES below being present in `answers`
    with the shapes the Decisions API returns (noul/score/choice/box).
    If a future prompt version renames or drops a question, decide()
    raises MalformedAnswers rather than crashing on a KeyError or
    silently computing a decision from partial data -- that failure
    mode is fail-SAFE (escalate), not fail-open (walk).

    Required question names: scene_assessable, hazard_proximity,
    hazard_type, crosswalk_present, signal_box_present,
    pedestrian_signal_color. (signal_box is captured by the prompt but
    not currently used here -- see README "known gaps".) Also reads
    `hazard_proximity.confidence` when present (optional -- older
    answers without it are treated as confidence=1.0, i.e. trusted).

    hazard_proximity confidence gate: a BLOCK triggered by the plain
    hp>=T_STOP fallback (not the moving_vehicle fast path, which has
    its own strong signal) is only honored when hazard_proximity's own
    reported confidence is >= HP_CONFIDENCE_FLOOR. Found by diagnosing
    liquid/d1-ultra's 3 false-BLOCK failures on ambiguous
    "other_obstacle" scenes (a yielding car misread as a generic
    obstacle): all 3 had hp confidence 0.00-0.11, while every verified
    TRUE BLOCK in the golden dataset (real stairs/car/tram hazards, both
    jev-1.14 and d1-ultra) had hp confidence >= 0.32. A low-confidence
    hp>=1.0 reading escalates to ESCALATE_UNCERTAIN instead of either
    BLOCKing (risk: false positive, erodes trust) or falling through to
    WALK (risk: the one thing this whole system exists to prevent).
    ESCALATE_UNCERTAIN is a new, distinct terminal state from
    ESCALATE_UNREADABLE: the camera/scene was fine, the model is just
    not sure what it's looking at.

    Returns one of: WALK, WAIT, BLOCK, ESCALATE_UNREADABLE, ESCALATE_UNCERTAIN.
    """
    try:
        hp = _require(answers, "hazard_proximity", "score")
        htype = _require(answers, "hazard_type", "choice")
        crosswalk_present = _require(answers, "crosswalk_present", "noul")
        sig_present = _require(answers, "signal_box_present", "noul")
        sig_color = _require(answers, "pedestrian_signal_color", "choice")
        assessable = _require(answers, "scene_assessable", "noul")
    except MalformedAnswers:
        # Fail SAFE, not fail open: a malformed/incomplete answer set
        # (schema drift from a prompt change, a truncated API response,
        # ...) must never silently fall through to a default WALK
        # computed from partial data. Treat it exactly like an
        # unreadable camera frame.
        return "ESCALATE_UNREADABLE"

    # hazard_proximity.confidence is optional in older prompt versions;
    # treat missing as fully trusted so this gate is a no-op on prompts
    # that don't report it, rather than silently blocking everything.
    hp_confidence = answers.get("hazard_proximity", {}).get("confidence", 1.0)
    if hp_confidence is None:
        hp_confidence = 1.0

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

    # Emergency check FIRST, ahead of the red-light WAIT below (logic
    # v1.2 reorder -- was after the red check in v1.1). A vehicle
    # actively moving into/through the path has NOT yielded, regardless
    # of what the pedestrian signal shows. Checking red before this let
    # "a car running the red light is entering the crossing" silently
    # downgrade from BLOCK to WAIT, which understates the urgency of an
    # active collision risk vs. a routine "wait for the light" message.
    if htype == "moving_vehicle" and hp >= T_STOP:
        return "BLOCK"

    if crossing_relevant and sig_color == "red":
        return "WAIT"

    # Marked crosswalk + (no signal OR green) -> walk. Per Austrian StVO
    # Sec9/Sec76: at an unsignalized crosswalk, vehicles are legally
    # required to yield; a stopped/yielding vehicle there is not a
    # hazard. Excludes all non-vehicle hazard types (stairs/rail/
    # obstacle) and moving_vehicle -- a crosswalk doesn't make those
    # safe. (moving_vehicle is already handled above and never reaches
    # here.)
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
        # Confidence gate: don't trust a near-threshold hazard score the
        # model itself flags as a coin-flip. See docstring above for the
        # evidence (d1-ultra's 3 false positives all had hp confidence
        # <=0.11; every true positive we've recorded had >=0.32).
        # Escalates rather than falls through to WALK -- ambiguity about
        # a possible hazard must never resolve to the most permissive
        # answer.
        if hp_confidence < HP_CONFIDENCE_FLOOR:
            return "ESCALATE_UNCERTAIN"
        return "BLOCK"

    return "WALK"


def evaluate_image_b64(image_b64: str, prompt_spec: dict, api_key: str) -> dict:
    """Full pipeline: one API call + decision. Returns a dict with
    decision, raw answers, latency, and cost.

    If the API call fails (non-200), decision is "ERROR" -- a transport
    failure, not a safety decision; callers must not treat this as
    WALK. If the call succeeds but `parsed["answers"]` is missing or the
    wrong shape entirely (not just one malformed field -- decide()
    already handles that case), decision falls back to
    ESCALATE_UNREADABLE rather than propagating the exception, for the
    same fail-safe-not-fail-open reason documented on decide().
    """
    status, lat, parsed, raw_body = call_jev(image_b64, prompt_spec, api_key)
    if status != 200:
        return {"decision": "ERROR", "http_status": status, "lat": lat, "raw_body": raw_body}
    try:
        decision = decide(parsed["answers"])
    except (KeyError, TypeError) as e:
        return {
            "decision": "ESCALATE_UNREADABLE",
            "decision_error": f"{type(e).__name__}: {e}",
            "answers": parsed.get("answers"),
            "lat": lat,
            "cost": parsed.get("usage", {}).get("cost"),
            "model": parsed.get("model"),
            "id": parsed.get("id"),
            "prompt_version": prompt_spec.get("version", "unknown"),
        }
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
