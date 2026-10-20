"""
Simulated streaming test: replays video1/video2's recorded L1 decisions
(from the live eval results already in eval/results.json or re-derived
live) through L2RollingBuffer, exactly as if frames were arriving from
a real camera stream one at a time.

This is the "loop until it fits a streaming video" harness -- run it,
inspect the flip count before/after L2, tune L2RollingBuffer's
parameters, re-run, until flicker is eliminated without masking a real
transition.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from l2_buffer import L2RollingBuffer


def simulate(frames: list[tuple[float, str]], window_s=3.0, deescalation_frames=3):
    """frames: list of (timestamp_s, raw_l1_decision). Returns
    (raw_flips, stabilized_flips, stabilized_sequence)."""
    buf = L2RollingBuffer(window_s=window_s, deescalation_frames=deescalation_frames)
    stabilized_seq = []
    prev_raw = None
    prev_stable = None
    raw_flips = 0
    stable_flips = 0
    for t, raw in frames:
        if prev_raw is not None and raw != prev_raw:
            raw_flips += 1
        prev_raw = raw

        stable = buf.push(t, raw)
        stabilized_seq.append((t, raw, stable))
        if prev_stable is not None and stable != prev_stable:
            stable_flips += 1
        prev_stable = stable

    return raw_flips, stable_flips, stabilized_seq


if __name__ == "__main__":
    # Known-good recorded L1 sequences from this session's live testing
    # (video1 = red/green crosswalk, 41 frames @ 0.5s; the real
    # single-frame flicker we proved exists this session).
    video1_raw_sequence = [
        (0.0, "WAIT"), (0.5, "WAIT"), (1.0, "WAIT"), (1.5, "WAIT"), (2.0, "WAIT"),
        (2.5, "WAIT"), (3.0, "WAIT"), (3.5, "WAIT"), (4.0, "WAIT"), (4.5, "WALK"),
        (5.0, "WAIT"), (5.5, "WAIT"), (6.0, "WALK"), (6.5, "WAIT"), (7.0, "WALK"),
        (7.5, "WALK"), (8.0, "WAIT"), (8.5, "WAIT"), (9.0, "WALK"), (9.5, "WALK"),
        (10.0, "WALK"), (10.5, "WALK"), (11.0, "WALK"), (11.5, "WALK"), (12.0, "WALK"),
        (12.5, "WAIT"), (13.0, "WALK"), (13.5, "WALK"), (14.0, "WAIT"), (14.5, "BLOCK"),
        (15.0, "WALK"), (15.5, "WALK"), (16.0, "WALK"), (16.5, "WALK"), (17.0, "WALK"),
        (17.5, "WALK"), (18.0, "WALK"), (18.5, "WALK"), (19.0, "WALK"), (19.5, "WALK"),
        (20.0, "WALK"),
    ]

    video2_raw_sequence = [
        (0.0, "WAIT"), (0.5, "WAIT"), (1.0, "WAIT"), (1.5, "WAIT"), (2.0, "WAIT"),
        (2.5, "WAIT"), (3.0, "WAIT"), (3.5, "WAIT"), (4.0, "WAIT"), (4.5, "WAIT"),
        (5.0, "WAIT"), (5.5, "WAIT"), (6.0, "WAIT"), (6.5, "WAIT"), (7.0, "WAIT"),
        (7.5, "WAIT"), (8.0, "WAIT"), (8.5, "WAIT"), (9.0, "WAIT"), (9.5, "WAIT"),
        (10.0, "WAIT"), (10.5, "WAIT"), (11.0, "WAIT"), (11.5, "WAIT"), (12.0, "WAIT"),
        (12.5, "WAIT"), (13.0, "WAIT"), (13.5, "WALK"), (14.0, "WAIT"), (14.5, "WAIT"),
        (15.0, "WAIT"), (15.5, "WAIT"), (16.0, "WAIT"), (16.5, "WAIT"), (17.0, "WAIT"),
        (17.5, "WAIT"), (18.0, "WALK"), (18.5, "WALK"), (19.0, "WALK"), (19.5, "WALK"),
        (20.0, "WALK"),
    ]

    for params in [
        dict(window_s=3.0, deescalation_frames=1),
        dict(window_s=3.0, deescalation_frames=2),
        dict(window_s=3.0, deescalation_frames=3),
        dict(window_s=3.0, deescalation_frames=4),
        dict(window_s=3.0, deescalation_frames=6),
        dict(window_s=3.0, deescalation_frames=8),
    ]:
        raw_flips, stable_flips, seq = simulate(video1_raw_sequence, **params)
        v1_tail_walk = seq[-1][2]  # must still correctly settle on WALK at the end
        raw_flips2, stable_flips2, seq2 = simulate(video2_raw_sequence, **params)
        v2_tail_walk = seq2[-1][2]
        print(f"params={params}: v1 raw={raw_flips} stable={stable_flips} tail={v1_tail_walk} "
              f"| v2 raw={raw_flips2} stable={stable_flips2} tail={v2_tail_walk}")

    print("\n--- Detailed trace at deescalation_frames=3 ---")
    raw_flips, stable_flips, seq = simulate(video1_raw_sequence, window_s=3.0, deescalation_frames=3)
    for t, raw, stable in seq:
        marker = "  <-- DIFFERS" if raw != stable else ""
        print(f"t={t:5.1f}s  raw={raw:8s} stable={stable:8s}{marker}")
    print(f"\nFinal: raw_flips={raw_flips}, stable_flips={stable_flips}")
