# VisionAeye

Hazard-detection pipeline for blind/low-vision pedestrians, built on the
TypeSafe `jev-1.14` Decisions API. Model perceives (typed, calibrated
questions over a single frame); code decides (WALK / WAIT / BLOCK) via
auditable thresholds in `pipeline/pediev_pipeline.py`.

**Current production version: PediJev v1.1** (locked). One model, one
API call per frame, no two-stage crop/recall. See `pipeline/README.md`
(below) for the exact prompt, decision logic, and validated results.

## Repo layout

```
data/
  images/              23 static test images (independently reviewed golden labels)
  golden_dataset.json  metadata for the 23 static images: scene description,
                        golden decision, hazard type, signal state, rationale,
                        and the model's own test result for each
  video1_frames/        41 frames (0.5s apart, 20.67s clip) -- first test video
  video2_frames/        41 frames (0.5s apart, 20.67s clip) -- second test video
pipeline/
  pediev_pipeline.py    the locked PediJev v1.1 pipeline: prompt + decision code
```

Note: the original `.mov` source files are not included (ephemeral cache,
expired before this export) -- only the extracted frames, which are what
every reported result in this repo was actually tested against.

## Results summary (PediJev v1.1)

- **Static dataset: 23/23 (100%)** on the independently-labeled images.
- Fixed a real safety bug found by manual review: an unconditional
  "if red signal -> WAIT" let an irrelevant distant light govern someone
  walking on a sidewalk. Fix gates the signal check behind whether a
  crossing is actually relevant to the current forward path.
- Video 1 (41 frames): stable, 1-3 flips depending on version tested
  (see `data/golden_dataset.json` entries / session history for detail).
- Video 2 (41 frames): stable, 1 flip (the one real red-to-green
  transition in the clip).
- Model: `jev-1.14` via OpenRouter's Decisions API
  (`https://openrouter.ai/api/alpha/decisions`), ~$0.00017/image,
  ~0.7-0.9s median latency per call.

## Running the pipeline

```bash
export OPENROUTER_API_KEY=...
python pipeline/pediev_pipeline.py data/images/img_f74679f233e2.jpg
```

Prints the full decision JSON: `decision`, raw model `answers` (hazard
proximity score/type, crosswalk/signal presence, signal color, and the
signal's bounding box), latency, and cost.

## Design principles

1. **Model perceives, code decides.** Prompts are neutral and atomic
   (no liability framing); every go/no-go threshold lives in
   `decide()`, not in the prompt -- reviewable and tunable without a
   model change.
2. **Single call, single model.** No cross-model hybrid, no Stage-2
   crop-and-recall. Every question (hazard, crosswalk, signal
   color/presence, and the signal's `box` location) is answered in one
   request.
3. **Safety-first fallbacks.** Ambiguous/unreadable states never
   silently resolve to WALK. An actively-moving vehicle always
   overrides the crosswalk-safe assumption. Austrian StVO Sec9/Sec76
   crosswalk-yielding rules are encoded directly in the decision logic.
