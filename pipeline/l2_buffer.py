"""
L2 -- short-term situational memory for the VisionAeye pipeline.

Problem this solves (proven empirically earlier in this project, not
hypothetical): a single L1 frame can flicker. On video1 (41 frames,
0.5s apart, an unchanging red light), raw per-frame hazard/signal
answers swung confidence 0.03->0.85 between ADJACENT frames with no
real-world change that fast. Feeding raw per-frame L1 decisions
straight to speech would announce "wait... walk... wait... walk" on a
light that never changed -- unusable and actively unsafe (trains the
user to ignore warnings).

Design, "model perceives, code decides" extended to time:
- L1 (pipeline/pediev_pipeline.py: decide()) still runs every frame,
  unchanged. L2 does NOT replace L1 or re-score hazards; it consumes
  L1's sequence of decisions over a rolling window and decides what the
  SPOKEN/arbitrated state should be.
- Safety-asymmetric smoothing: escalating to a more urgent state
  (WALK -> WAIT/BLOCK) is INSTANT, on the very first frame that reports
  it. De-escalating (WAIT/BLOCK -> WALK) requires the less urgent state
  to persist for N consecutive frames before being trusted. This is
  deliberate: a missed hazard is far worse than a slightly late "all
  clear" -- the asymmetry mirrors the "better unsure than confidently
  wrong" principle already used in decide()'s confidence gate.
- ESCALATE_* states are treated as urgent (same severity as WAIT/BLOCK)
  for the purposes of the hysteresis -- consistent with how
  pipeline/arbitration.py treats them.

This module is deliberately a rolling buffer + a pure stateful
decision function -- no model calls, no I/O, so it is unit-testable
without any live dependency, and ready to be fed either live camera
frames (via L1 calls per frame) or the recorded video-frame sequences
already in data/video1_frames, data/video2_frames for regression
testing.
"""
from collections import deque
from dataclasses import dataclass, field


URGENT = {"WAIT", "BLOCK", "ESCALATE_UNREADABLE", "ESCALATE_UNCERTAIN"}

# Severity ranking for escalation-vs-de-escalation comparisons. Higher
# number = more urgent. Ties (e.g. the two ESCALATE_* states) are
# treated as equally urgent to each other.
_SEVERITY = {
    "WALK": 0,
    "WAIT": 1,
    "ESCALATE_UNCERTAIN": 1,
    "ESCALATE_UNREADABLE": 1,
    "BLOCK": 2,
}


def _severity(decision: str) -> int:
    return _SEVERITY.get(decision, 2)  # unknown decisions treated as max-urgent, fail-safe


@dataclass
class FrameRecord:
    """One L1 result in the rolling window."""
    timestamp_s: float
    decision: str


@dataclass
class L2RollingBuffer:
    """Maintains a rolling window of L1 decisions and produces a
    de-flickered 'stabilized' decision for arbitration/speech.

    window_s: how many seconds of history to retain (default 3.0s,
      per the product ask -- "reason the last 3 seconds").
    deescalation_frames: how many consecutive LESS URGENT frames are
      required before accepting a de-escalation (e.g. BLOCK -> WALK).
      Tuned empirically against video1/video2 below; default 3
      consecutive frames at the project's 0.5s sampling rate = 1.5s of
      sustained agreement before trusting "it's safe now."
    """

    window_s: float = 3.0
    deescalation_frames: int = 3
    _history: deque = field(default_factory=deque)
    _stable_decision: str = "WALK"
    _deescalation_run: int = 0
    _deescalation_candidate: str | None = None

    def push(self, timestamp_s: float, decision: str) -> str:
        """Feed one new L1 decision. Returns the STABILIZED decision to
        hand to arbitration/speech for this tick (may differ from the
        raw `decision` passed in -- that's the whole point)."""
        self._history.append(FrameRecord(timestamp_s, decision))
        while self._history and timestamp_s - self._history[0].timestamp_s > self.window_s:
            self._history.popleft()

        cur_severity = _severity(self._stable_decision)
        new_severity = _severity(decision)

        if new_severity >= cur_severity:
            # Escalation (or same severity) is instant, no debounce.
            # A rising hazard must never be delayed by smoothing.
            self._stable_decision = decision
            self._deescalation_run = 0
            self._deescalation_candidate = None
            return self._stable_decision

        # De-escalation candidate (new_severity < cur_severity): require
        # N consecutive matching, less-urgent frames before trusting it.
        if decision == self._deescalation_candidate:
            self._deescalation_run += 1
        else:
            self._deescalation_candidate = decision
            self._deescalation_run = 1

        if self._deescalation_run >= self.deescalation_frames:
            self._stable_decision = decision
            self._deescalation_run = 0
            self._deescalation_candidate = None

        return self._stable_decision

    @property
    def history(self) -> list:
        return list(self._history)

    @property
    def stable_decision(self) -> str:
        return self._stable_decision

    def reset(self) -> None:
        self._history.clear()
        self._stable_decision = "WALK"
        self._deescalation_run = 0
        self._deescalation_candidate = None
