# VisionAeye Roadmap

Full product vision: an always-on AI agent running on smart glasses that
helps a blind or low-vision person navigate the world safely AND
interact with it -- reading a restaurant menu, finding a product in a
grocery store, describing surroundings on request.

## Architecture: three layers, not two

- **L1 -- Reflex safety** (built, this repo's `pipeline/`). Per-frame,
  sub-second, safety-only, never blocked by anything else. Current
  state: PediJev v1.1, 35/35 on the golden dataset, locked.
- **L2 -- Short-term situational memory** (not started). Reasons over a
  rolling ~3s window of frames instead of one snapshot -- smooths
  single-frame flicker, understands trajectory ("this car is slowing
  down" vs "a car exists somewhere in frame").
- **L3 -- Deliberate assistant** (not started). User-triggered,
  latency/cost tolerant. Menu reading, product finding, wayfinding.

**The hard problem that makes this one system instead of three demos:
arbitration.** L1 must always be able to interrupt L3 instantly, with
zero exceptions, or a real hazard gets missed because the glasses were
mid-sentence about soup.

## Epics, in priority order

### Epic 4 -- Arbitration layer (BUILT)
L1-always-wins interrupt contract. `pipeline/arbitration.py` +
`pipeline/test_arbitration.py` (16 tests). Built against a stub L3
before any real L2/L3 model exists, on purpose -- this is the piece
most likely to be built badly if retrofitted onto a shipped
conversational feature later.

Still open: real audio-output wiring (this is pure logic right now, no
TTS/speaker integration), and a decision on debounce if re-announcing
every tick during a sustained hazard proves annoying in practice
(deliberately NOT solved at this layer -- see code comments).

### Epic 2 -- Short-term situational memory (L2) -- BUILT
`pipeline/l2_buffer.py` (rolling buffer + hysteresis) + `pipeline/stream.py`
(wires L1 -> L2 -> arbitration). 16 unit tests
(`pipeline/test_l2_buffer.py`) including replays of the exact flicker
patterns recorded from this project's live video testing. Tuned via
`pipeline/simulate_l2_stream.py` against real recorded sequences, then
verified end-to-end against the real video frame files
(`pipeline/real_e2e_test.py`), not just simulated decisions.

What shipped: safety-asymmetric hysteresis (instant escalation, 3-frame
debounced de-escalation) over a 3-second rolling window. This answers
item 4 from the original Epic 2 plan (L2->L1 relationship: L2 never
overrides a BLOCK/WAIT with less history than it takes to trust a
WALK) but NOT the others:

Still open (not built):
1. L2 currently only smooths L1's categorical decision (WALK/WAIT/BLOCK)
   over time -- it does NOT yet reason over raw multi-frame visual
   content (trajectory of a vehicle, "is this person now stationary").
   That needs either a true video-input model or a richer per-frame
   feature (e.g. feeding L2 the hazard_proximity score, not just the
   final decision) -- scoped but not implemented.
2. Video-input vs sliding-window model evaluation (item 2) -- not done.
3. Golden dataset v2 with multi-frame ground truth (item 5) -- not done;
   current golden dataset is still L1-only single frames.
4. Extending the eval/run_eval.py pattern to L2 (item 6) -- not done;
   L2 is validated by `test_l2_buffer.py`'s deterministic unit tests
   and the one real end-to-end video replay, not a golden-dataset CI
   job yet.

### Epic 7 -- Evaluation & reliability infra (ONGOING, extend per epic)
Already built for L1: golden dataset, CI, confidently-wrong vs
honestly-escalated distinction. Each new epic needs its own golden
dataset and the same rigor -- do not skip this the way L1's early
iterations did (the confidently-wrong/escalated distinction was a
retrofit, not built in from day one).

### Epic 3 -- Always-on streaming pipeline
Edge vs cloud split, power/bandwidth budget, adaptive frame rate,
**privacy design (continuous video of bystanders, receipts, card
numbers in a store -- GDPR-relevant in Austria/EU, must be decided
before building, not after)**.

### Epic 5 -- Conversational assistant (L3) -- PARTIALLY BUILT
`pipeline/l3_assistant.py` (vision-chat call, `ask_about_image()`) +
`VisionAeyeStream.ask()` (wires L3 into arbitration -- queues during an
active hazard, resumes after it clears, speaks an honest failure
message on a transport error rather than going silent). 10 tests
(6 mocked unit tests + 4 integration tests proving the queue-during-
hazard safety case). Verified live against a real image.

Still open (not built):
1. Wake word / voice trigger -- `ask()` currently takes the question as
   a plain string argument; there's no speech-to-text or activation
   mechanism wired in yet.
2. OCR-specific tuning for small/dense text (a printed menu font vs a
   general photo) -- current prompt is a generic vision-chat
   instruction, not validated against a real menu photo yet (no menu
   test image was available this session -- needs a real one from the
   user to validate this specific use case, not just "can it see a
   street scene").
3. Multi-turn dialogue ("tell me more about the pasta") -- each `ask()`
   call is stateless/one-shot today, no conversation history.
4. Cost/rate limiting -- nothing currently stops a user from spamming
   `ask()`; needs an explicit trigger debounce before this goes near
   real hardware.

### Epic 6 -- Spatial guidance / wayfinding
Named-target object detection + localization (different, harder
problem than generic hazard detection), spatial language generation
("left side, two shelves down"), likely needs depth estimation since a
bounding box alone doesn't give reliable distance. Likely needs a
bigger/different model than jev-1.14 -- scope a fresh model eval here,
don't assume the current stack extends.

### Epic 8 -- Hardware integration
Camera/compute/battery given the edge/cloud split, audio hardware
choice. Longest lead time -- prototype on a phone-mounted camera first.

## Sequencing notes

- Build arbitration (Epic 4) before any real L3, not after -- done.
- L2 (Epic 2) is next since it directly extends what's proven (L1) and
  has no hardware/privacy dependencies blocking it.
- Epic 3 (streaming) and Epic 8 (hardware) can run in parallel with
  software work, phone-camera-prototype first.
- Resist building Epic 5/6 (the most demo-able parts) before
  arbitration and L2 exist -- a live demo where a hazard interrupts a
  menu-read badly would undercut the whole pitch.
