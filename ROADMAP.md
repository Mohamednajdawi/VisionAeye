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

### Epic 2 -- Short-term situational memory (L2) -- NEXT
1. Define the L2 output contract: what "bigger picture" means
   concretely (hazard trajectory, signal stability) vs L1's instant
   per-frame answer.
2. Evaluate true video-input models vs a sliding-window-of-frames fed
   to a single-image model.
3. Build a rolling frame buffer feeding L2 without re-architecting L1's
   per-frame loop.
4. Define L2->L1 relationship: L2 should never override a BLOCK/WAIT,
   only add context or smooth a borderline WALK.
5. Golden dataset v2: short video clips with multi-frame ground truth
   (current dataset is L1-only, single frames).
6. Extend `eval/run_eval.py`'s pattern to L2.

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

### Epic 5 -- Conversational assistant (L3)
Wake word/trigger, OCR + scene-text reading (menu reading is mostly
this), one-shot Q&A first, multi-turn dialogue later. Explicit
user-initiated trigger only -- L3 must not run continuously (cost).

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
