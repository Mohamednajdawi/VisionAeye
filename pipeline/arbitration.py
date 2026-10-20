"""
Arbitration engine for the VisionAeye smart-glasses assistant.

Three layers feed audio output:
  L1 = reflex safety (PediJev, pipeline/pediev_pipeline.py) -- per-frame,
       always-on, produces WALK/WAIT/BLOCK/ESCALATE_*.
  L2 = short-term situational memory (not yet built) -- rolling few-second
       window, smooths L1 flicker, adds context. Out of scope here.
  L3 = deliberate assistant (not yet built) -- user-triggered, slower,
       answers "what's on the menu" / "where is the milk".

THE RULE THIS MODULE EXISTS TO ENFORCE: L1 always wins. If L3 is
mid-sentence describing a menu and L1 reports WAIT or BLOCK, L3's audio
is interrupted immediately, every time, with zero exceptions. A WALK
from L1 is silent (no audio) -- we do not nag the user on every clear
frame, only on an actual change in hazard state.

This is built BEFORE any real L2/L3 model exists, against a stub
interface, specifically so the interrupt contract exists from day one
and isn't retrofitted onto a shipped conversational assistant later.
"""
from dataclasses import dataclass, field
from enum import Enum


class L3State(Enum):
    IDLE = "idle"        # nothing playing
    SPEAKING = "speaking"  # L3 audio actively playing
    PAUSED = "paused"     # L3 audio paused mid-utterance by an active hazard


# Decisions that must interrupt L3 and speak immediately. Mirrors the
# ESCALATE_* additions in pipeline/pediev_pipeline.py's decide() --
# an escalation is exactly as urgent as a BLOCK, since both mean
# "do not trust that the path is safe right now."
URGENT_DECISIONS = {"WAIT", "BLOCK", "ESCALATE_UNREADABLE", "ESCALATE_UNCERTAIN"}


@dataclass
class AudioAction:
    """One instruction for the audio output layer. `kind` is one of:
    SPEAK_URGENT (interrupt whatever is playing, speak now, highest
    priority), INTERRUPT (stop L3 audio immediately, no new speech
    yet -- used internally, callers normally see SPEAK_URGENT instead),
    SPEAK (normal L3 speech, only if no hazard is active), QUEUE (an L3
    request arrived while a hazard is active -- do not speak, remember
    it for later), RESUME (hazard cleared, continue L3 from where it
    paused or start the queued request)."""
    kind: str
    text: str = ""


@dataclass
class ArbitrationEngine:
    """Pure state machine, no I/O, no model calls -- feed it L1
    decisions and L3 requests, it tells you what audio action to take.
    Fully deterministic and unit-testable without any live dependency."""

    l3_state: L3State = L3State.IDLE
    active_hazard: bool = False
    paused_text: str | None = None   # what L3 was saying when interrupted
    pending_text: str | None = None  # an L3 request that arrived during a hazard

    def on_l1_decision(self, decision: str) -> list[AudioAction]:
        """Feed the latest L1 decision (WALK/WAIT/BLOCK/ESCALATE_*).
        Call this every frame/tick, including every WALK -- the engine
        only emits audio actions on a STATE CHANGE (hazard starting or
        clearing), not on every repeated identical decision."""
        urgent = decision in URGENT_DECISIONS
        actions: list[AudioAction] = []

        if urgent and not self.active_hazard:
            # Hazard just started.
            self.active_hazard = True
            if self.l3_state == L3State.SPEAKING:
                self.l3_state = L3State.PAUSED
                actions.append(AudioAction("INTERRUPT"))
            actions.append(AudioAction("SPEAK_URGENT", decision))

        elif urgent and self.active_hazard:
            # Hazard is ongoing and/or changed severity (e.g. WAIT -> BLOCK).
            # Re-announce so the user always hears the CURRENT state, not
            # a stale one -- silence during an ongoing hazard is unsafe.
            actions.append(AudioAction("SPEAK_URGENT", decision))

        elif not urgent and self.active_hazard:
            # Hazard cleared.
            self.active_hazard = False
            if self.l3_state == L3State.PAUSED:
                self.l3_state = L3State.SPEAKING
                actions.append(AudioAction("RESUME", self.paused_text or ""))
                self.paused_text = None
            elif self.pending_text is not None:
                self.l3_state = L3State.SPEAKING
                actions.append(AudioAction("RESUME", self.pending_text))
                self.pending_text = None

        # not urgent and no active hazard: steady-state WALK, no action.
        return actions

    def on_l3_request(self, text: str) -> list[AudioAction]:
        """L3 has something to say (a completed answer to a user query,
        or the next chunk of an ongoing one). If a hazard is currently
        active, this is queued, never spoken over the hazard."""
        if self.active_hazard:
            self.pending_text = text
            return [AudioAction("QUEUE", text)]
        self.l3_state = L3State.SPEAKING
        self.paused_text = text
        return [AudioAction("SPEAK", text)]

    def on_l3_done(self) -> None:
        """L3 finished speaking its current utterance normally (not
        interrupted). Call this when playback completes."""
        if self.l3_state == L3State.SPEAKING:
            self.l3_state = L3State.IDLE
            self.paused_text = None
