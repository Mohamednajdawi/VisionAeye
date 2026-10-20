"""
Unit tests for L2RollingBuffer (pipeline/l2_buffer.py).

Parameters (window_s=3.0, deescalation_frames=3) were chosen by
iterating pipeline/simulate_l2_stream.py against the two real recorded
video sequences from this project's earlier live testing until flicker
was eliminated without masking either video's real transition -- see
that file and ROADMAP.md Epic 2 for the tuning log. These tests lock
that behavior in so a future change can't silently regress it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from l2_buffer import L2RollingBuffer  # noqa: E402


class TestInstantEscalation(unittest.TestCase):
    """A rising hazard must never be delayed by smoothing -- escalation
    is instant, on the very first frame that reports it."""

    def test_walk_to_wait_is_instant(self):
        buf = L2RollingBuffer()
        self.assertEqual(buf.push(0.0, "WALK"), "WALK")
        self.assertEqual(buf.push(0.5, "WAIT"), "WAIT")

    def test_walk_to_block_is_instant(self):
        buf = L2RollingBuffer()
        buf.push(0.0, "WALK")
        self.assertEqual(buf.push(0.5, "BLOCK"), "BLOCK")

    def test_wait_to_block_escalation_is_instant(self):
        """Severity increase within 'urgent' states (WAIT -> BLOCK) is
        also instant -- same reasoning as arbitration's reannouncement
        on severity change."""
        buf = L2RollingBuffer()
        buf.push(0.0, "WAIT")
        self.assertEqual(buf.push(0.5, "BLOCK"), "BLOCK")

    def test_single_frame_hazard_is_never_suppressed(self):
        """Even a lone hazard frame surrounded by WALK must surface
        immediately -- the de-escalation debounce only delays the
        return to WALK, never the hazard itself."""
        buf = L2RollingBuffer()
        buf.push(0.0, "WALK")
        self.assertEqual(buf.push(0.5, "BLOCK"), "BLOCK")


class TestDeescalationDebounce(unittest.TestCase):
    """De-escalating (WAIT/BLOCK -> WALK) requires N consecutive
    matching less-urgent frames -- this is what eliminates flicker."""

    def test_single_walk_frame_does_not_deescalate(self):
        buf = L2RollingBuffer(deescalation_frames=3)
        buf.push(0.0, "WAIT")
        self.assertEqual(buf.push(0.5, "WALK"), "WAIT")  # 1st WALK, not enough

    def test_two_walk_frames_do_not_deescalate(self):
        buf = L2RollingBuffer(deescalation_frames=3)
        buf.push(0.0, "WAIT")
        buf.push(0.5, "WALK")
        self.assertEqual(buf.push(1.0, "WALK"), "WAIT")  # 2nd WALK, still not enough

    def test_three_consecutive_walk_frames_deescalates(self):
        buf = L2RollingBuffer(deescalation_frames=3)
        buf.push(0.0, "WAIT")
        buf.push(0.5, "WALK")
        buf.push(1.0, "WALK")
        self.assertEqual(buf.push(1.5, "WALK"), "WALK")  # 3rd WALK -> trust it

    def test_interrupted_deescalation_run_resets(self):
        """A single flicker-back-to-WAIT mid-run must reset the
        consecutive count -- this is the actual flicker-elimination
        mechanism, proven against real recorded video1 data where raw
        WALK/WAIT swapped every 0.5-1s."""
        buf = L2RollingBuffer(deescalation_frames=3)
        buf.push(0.0, "WAIT")
        buf.push(0.5, "WALK")   # run=1
        buf.push(1.0, "WALK")   # run=2
        self.assertEqual(buf.push(1.5, "WAIT"), "WAIT")  # flicker back -> reset, still WAIT
        buf.push(2.0, "WALK")   # run=1 again (restarted)
        buf.push(2.5, "WALK")   # run=2
        self.assertEqual(buf.push(3.0, "WALK"), "WALK")  # run=3 -> now trusted


class TestRealVideoRegressions(unittest.TestCase):
    """Replays the exact flicker patterns recorded from this project's
    live video testing (video1: red/green crosswalk, 12 raw flips in
    41 frames; video2: held red light, 3 raw flips) and asserts L2
    eliminates flicker while preserving every real transition."""

    VIDEO1_SEQUENCE = [
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

    VIDEO2_SEQUENCE = (
        [(round(i * 0.5, 1), "WAIT") for i in range(27)]  # 0.0 .. 13.0, held red
        + [(13.5, "WALK")]                                  # single-frame flicker
        + [(round(i * 0.5, 1), "WAIT") for i in range(28, 36)]  # 14.0 .. 17.5, back to held red
        + [(18.0, "WALK"), (18.5, "WALK"), (19.0, "WALK"), (19.5, "WALK"), (20.0, "WALK")]  # real transition
    )

    def _run(self, sequence):
        buf = L2RollingBuffer(window_s=3.0, deescalation_frames=3)
        stable_seq = []
        for t, raw in sequence:
            stable_seq.append(buf.push(t, raw))
        flips = sum(1 for a, b in zip(stable_seq, stable_seq[1:]) if a != b)
        return stable_seq, flips

    def test_video1_flicker_eliminated(self):
        stable_seq, flips = self._run(self.VIDEO1_SEQUENCE)
        # Raw had 12 flips; stabilized must be well below that.
        self.assertLessEqual(flips, 4)

    def test_video1_ends_on_walk(self):
        stable_seq, _ = self._run(self.VIDEO1_SEQUENCE)
        self.assertEqual(stable_seq[-1], "WALK")

    def test_video1_moving_vehicle_event_preserved(self):
        """The real BLOCK event (moving vehicle, t=14.5s) must survive
        stabilization -- this is the thing L2 must never hide."""
        stable_seq, _ = self._run(self.VIDEO1_SEQUENCE)
        timestamps = [t for t, _ in self.VIDEO1_SEQUENCE]
        idx = timestamps.index(14.5)
        self.assertEqual(stable_seq[idx], "BLOCK")

    def test_video2_single_frame_flicker_absorbed(self):
        """The lone WALK blip at t=13.5s (surrounded by held WAIT) must
        NOT flip the stabilized output -- this is the exact flicker
        case L2 exists to fix."""
        stable_seq, _ = self._run(self.VIDEO2_SEQUENCE)
        timestamps = [t for t, _ in self.VIDEO2_SEQUENCE]
        idx = timestamps.index(13.5)
        self.assertEqual(stable_seq[idx], "WAIT")

    def test_video2_real_end_transition_preserved(self):
        """The real, sustained red->green transition at the end (5
        consecutive WALK frames) must still register -- flicker
        elimination must not come at the cost of masking a genuine
        change (this was an actual failure mode found while tuning:
        deescalation_frames>=6 masked this exact transition)."""
        stable_seq, _ = self._run(self.VIDEO2_SEQUENCE)
        self.assertEqual(stable_seq[-1], "WALK")

    def test_video2_flicker_eliminated(self):
        stable_seq, flips = self._run(self.VIDEO2_SEQUENCE)
        self.assertLessEqual(flips, 1)  # only the one real transition


class TestWindowExpiry(unittest.TestCase):
    def test_old_history_is_pruned(self):
        buf = L2RollingBuffer(window_s=3.0, deescalation_frames=3)
        buf.push(0.0, "WAIT")
        buf.push(10.0, "WALK")  # 10s later, way outside the 3s window
        self.assertEqual(len(buf.history), 1)  # the stale WAIT entry is pruned


class TestReset(unittest.TestCase):
    def test_reset_clears_state(self):
        buf = L2RollingBuffer()
        buf.push(0.0, "BLOCK")
        buf.reset()
        self.assertEqual(buf.stable_decision, "WALK")
        self.assertEqual(len(buf.history), 0)


if __name__ == "__main__":
    unittest.main()
