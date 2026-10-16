"""
PediJev v1.1 -- production hazard-detection pipeline for a blind-pedestrian
smart-glasses prototype. Single model (jev-1.14 via OpenRouter Decisions
API), single call per frame. See README.md for full context and results.
"""
import json, os, time
import urllib.request, urllib.error

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "jev-1.14"

STATE_TEXT = (
    "Single frame from a chest-height camera worn by a blind or low-vision "
    "pedestrian, facing their walking direction. The 'forward path' is the "
    "strip of ground directly ahead: about shoulder width (~1 m) and about "
    "5 m long. In the image it runs from the bottom-center of the frame "
    "toward the vanishing point. These answers trigger spoken warnings, so "
    "missing a hazard is far worse than a false alarm."
)

QUESTIONS = {
    "scene_assessable": {
        "type": "noul",
        "instructions": (
            "Is this photo usable for judging the scene? Answer no only for "
            "camera problems: most of the frame is motion-blurred, too dark, "
            "overexposed, or covered by a hand, finger, or clothing. Answer "
            "yes if the photo is clear, even when a vehicle, person, or "
            "object blocks the view of the ground."
        ),
    },
    "hazard_proximity": {
        "type": "score",
        "instructions": (
            "How directly does a physical hazard (vehicle, step or drop, "
            "rail or track, or obstacle) intrude into the forward path? "
            "Judge only the forward path, not the whole scene. Ignore "
            "traffic lights and signs. For a moving vehicle, judge where it "
            "is heading, not only where it is now. If unsure between two "
            "levels, choose the higher one."
        ),
        "criteria": [
            "clear: Nothing is in or entering the forward path. Any hazards "
            "are clearly off to the side or more than about 5 m ahead.",
            "caution: A hazard is at the edge of the forward path or about "
            "2 to 5 m ahead, or a vehicle is moving toward the path but is "
            "not in it yet. The person can continue with care.",
            "stop: A hazard occupies or crosses the forward path within "
            "about 2 m (one or two steps). The person must stop or change "
            "direction now.",
        ],
    },
    "hazard_type": {
        "type": "choice",
        "instructions": (
            "What physical hazard, if any, is in or entering the forward "
            "path? Look at the scene first; answer none when the path is "
            "open. Only if two or more different hazards are clearly in the "
            "path, pick the more dangerous one, with a moving vehicle above "
            "any other type."
        ),
        "options": [
            "none", "moving_vehicle", "stairs_or_drop", "rail_or_track",
            "stopped_vehicle", "other_obstacle",
        ],
        "criteria": {
            "none": "Nothing in or entering the forward path.",
            "moving_vehicle": (
                "A car, bus, tram, bike, or scooter in motion that is in, "
                "crossing, or heading into the forward path."
            ),
            "stairs_or_drop": (
                "Actual stairs, a platform edge, a ledge, or a raised step "
                "of at least ankle height in the forward path. A lowered "
                "or flush curb at a marked crossing, painted crossing "
                "stripes, and flat road surface are not drops."
            ),
            "rail_or_track": (
                "Tram rails or train tracks running across or along the "
                "forward path."
            ),
            "stopped_vehicle": (
                "A parked or waiting vehicle that occupies or partly "
                "blocks the forward path."
            ),
            "other_obstacle": (
                "Any other physical object that blocks or narrows the "
                "forward path, such as a pole, bin, sign, barrier, "
                "bollard, or person."
            ),
        },
    },
    "crosswalk_present": {
        "type": "noul",
        "instructions": (
            "Is the forward path a marked pedestrian crosswalk (zebra "
            "stripes, a raised/colored crossing, or similar official "
            "crossing markings)? Answer yes even if a vehicle or signal is "
            "also present."
        ),
    },
    "signal_box_present": {
        "type": "noul",
        "instructions": (
            "Is a dedicated pedestrian crossing signal visible anywhere in "
            "the image? It is a small rectangular light box mounted on a "
            "pole, distinct from the large vehicle traffic lights. Answer "
            "yes even if it is small or distant and you cannot read its "
            "symbol or tell which light is lit."
        ),
    },
    "pedestrian_signal_color": {
        "type": "choice",
        "instructions": (
            "If a pedestrian crossing signal is visible, what color/symbol "
            "is it currently showing? If no signal is visible, or it is "
            "visible but you genuinely cannot tell the color (e.g. too "
            "small, glare, angle), answer 'unclear'."
        ),
        "options": ["red", "green", "unclear"],
        "criteria": {
            "red": (
                "The signal shows a lit red light or a red/orange standing "
                "human figure ('do not walk')."
            ),
            "green": (
                "The signal shows a lit green light or a green/white "
                "walking human figure ('walk')."
            ),
            "unclear": (
                "No pedestrian signal is visible, or one is visible but "
                "its color cannot be determined."
            ),
        },
    },
    "signal_box": {
        "type": "box",
        "instructions": (
            "the pedestrian crossing signal box (small rectangular light "
            "box on a pole, showing a walking or standing human figure "
            "icon), if visible anywhere in the image"
        ),
    },
}

T_STOP = 1.0


def call_jev(image_b64: str, timeout: int = 60):
    """One Decisions API call. Returns (status, latency_s, parsed_json_or_None, raw_body)."""
    payload = {
        "model": MODEL,
        "state": [
            {"type": "text", "text": STATE_TEXT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
        "questions": QUESTIONS,
    }
    req = urllib.request.Request(
        DECISIONS_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",  # required for image + box questions
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        status = e.code
    lat = round(time.time() - t0, 3)
    parsed = json.loads(body) if status == 200 else None
    return status, lat, parsed, body


def decide(answers: dict) -> str:
    """
    PediJev v1.1 decision logic. Model perceives (via `answers`), code
    decides. All thresholds live here, not in the prompt.

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


def evaluate_image_b64(image_b64: str):
    """Full pipeline: one API call + decision. Returns a dict with
    decision, raw answers, latency, and cost."""
    status, lat, parsed, raw_body = call_jev(image_b64)
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
    }


if __name__ == "__main__":
    import sys
    import base64

    if len(sys.argv) != 2:
        print("Usage: python pediev_pipeline.py <path_to_jpg>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    result = evaluate_image_b64(b64)
    print(json.dumps(result, indent=2))
