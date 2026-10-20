"""
End-to-end streaming pipeline: L1 (per-frame hazard decision) -> L2
(rolling-window de-flicker) -> arbitration (L1-always-interrupts-L3).

This is the "fits a streaming video" integration point: feed it one
frame at a time (as base64 JPEG + a timestamp) and it returns the
audio actions to take for that tick. Everything underneath
(pediev_pipeline.decide(), l2_buffer.L2RollingBuffer,
arbitration.ArbitrationEngine) is already unit-tested in isolation;
this class just wires them together with no new decision logic of its
own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pediev_pipeline import evaluate_image_b64, load_prompt, DEFAULT_PROMPT_PATH  # noqa: E402
from l2_buffer import L2RollingBuffer  # noqa: E402
from arbitration import ArbitrationEngine, AudioAction  # noqa: E402


class VisionAeyeStream:
    """One instance per active session (one pair of glasses, one
    continuous wear session). Not thread-safe -- call `on_frame`
    sequentially, one frame at a time, in timestamp order."""

    def __init__(self, prompt_path: str = DEFAULT_PROMPT_PATH, api_key: str = None,
                 l2_window_s: float = 3.0, l2_deescalation_frames: int = 3):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.prompt_spec = load_prompt(prompt_path)
        self.l2 = L2RollingBuffer(window_s=l2_window_s, deescalation_frames=l2_deescalation_frames)
        self.arbitration = ArbitrationEngine()

    def on_frame(self, image_b64: str, timestamp_s: float) -> dict:
        """Process one camera frame. Returns a dict with the raw L1
        result, the L2-stabilized decision, and the audio actions (if
        any) arbitration decided to take this tick."""
        l1_result = evaluate_image_b64(image_b64, self.prompt_spec, self.api_key)
        raw_decision = l1_result["decision"]

        stabilized = self.l2.push(timestamp_s, raw_decision)

        audio_actions = self.arbitration.on_l1_decision(stabilized)

        return {
            "timestamp_s": timestamp_s,
            "l1_raw_decision": raw_decision,
            "l1_result": l1_result,
            "l2_stabilized_decision": stabilized,
            "audio_actions": audio_actions,
        }

    def on_user_request(self, text: str) -> list:
        """A completed L3 answer (menu text, product location, etc.)
        ready to be queued for speech. Returns the audio actions."""
        return self.arbitration.on_l3_request(text)

    def on_l3_speech_done(self) -> None:
        self.arbitration.on_l3_done()
