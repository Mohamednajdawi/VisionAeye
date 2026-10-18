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
prompts/
  pediev_v1_1.json       the model-facing prompt as pure data: model id,
                          state text, and typed questions. No code. Swap or
                          A/B test prompt versions by adding a new file here
                          and pointing --prompt at it -- the pipeline never
                          needs to change.
pipeline/
  pediev_pipeline.py    model-agnostic API client + the decision engine
                          (decide()). Loads a prompt file at runtime; every
                          go/no-go threshold that turns model answers into
                          WALK/WAIT/BLOCK lives here in code, not in the
                          prompt -- reviewable and testable independently of
                          prompt wording.
```

### Why the prompt and the pipeline are split

- **The prompt is config, not code.** `prompts/pediev_v1_1.json` is pure
  JSON: model id, state text, typed questions. It has no logic in it.
  Changing a question's wording, adding a new question, or testing a
  different model entirely is a data change, with no risk of touching
  the decision logic.
- **The decision engine is the one place that can be unsafe or unsafe-fixed.**
  `decide()` in `pipeline/pediev_pipeline.py` owns every WALK/WAIT/BLOCK
  threshold. It is unit-testable against recorded model answers without
  calling the API, and it is the one function that needs review when a
  safety bug like the sidewalk/signal issue (see commit history) is
  found or fixed.
- **Coupling is explicit, not hidden.** `decide()` documents the exact
  question names it depends on (`hazard_proximity`, `hazard_type`,
  `crosswalk_present`, `signal_box_present`, `pedestrian_signal_color`).
  A new prompt version that renames or drops one of these needs a
  matching update to `decide()` -- this is a known, visible dependency,
  not a silent one.

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
# or explicitly pick a prompt version:
python pipeline/pediev_pipeline.py data/images/img_f74679f233e2.jpg --prompt prompts/pediev_v1_1.json
```

Prints the full decision JSON: `decision`, raw model `answers` (hazard
proximity score/type, crosswalk/signal presence, signal color, and the
signal's bounding box), latency, and cost.

## CI/CD: automated prompt evaluation

`.github/workflows/eval.yml` runs `eval/run_eval.py` automatically on
every push or PR that touches `prompts/`, `pipeline/`, or the golden
dataset. It evaluates every prompt file in `prompts/` against the full
golden dataset using the exact same `decide()` logic as production, and
publishes a results table to the GitHub Actions summary (and as a PR
comment, on pull requests).

Requires an `OPENROUTER_API_KEY` repo secret (Settings -> Secrets and
variables -> Actions). Run it manually from the Actions tab
(workflow_dispatch) with an optional `fail_under` accuracy threshold to
gate merges; the default is report-only (never fails the build) since a
new prompt is often deliberately exploratory before it's ready to be a
gate.

```bash
# Local run, same harness CI uses
export OPENROUTER_API_KEY=...
python eval/run_eval.py                      # evaluate every prompt in prompts/
python eval/run_eval.py --prompt prompts/pediev_v1_2.json
python eval/run_eval.py --fail-under 95       # exit 1 if any prompt scores below 95%
```

### Evaluated prompt versions

| Version | Model | Accuracy on golden dataset (23 images) |
|---|---|---|
| `pediev_v1_1.json` | jev-1.14 | 23/23 (100%) -- **current production** |
| `pediev_v1_2.json` | liquid/d1-ultra | 19/23 (82.6%) -- evaluated, not promoted (regresses on 3 previously-correct images plus the known boundary case) |



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
