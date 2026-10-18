"""
Unit tests for the PediJev decision engine (decide()).

Deliberately API-free: every test constructs a mocked `answers` dict in
the exact shape the Decisions API returns and asserts the decision.
This is what separates "is our logic correct" (deterministic, tested
here) from "is the model's perception good today" (noisy, tested by
eval/run_eval.py against live API calls) -- conflating the two was a
known gap; this file is the fix.

Run: python -m pytest pipeline/test_pediev_pipeline.py -v
  or: python pipeline/test_pediev_pipeline.py   (falls back to unittest)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pediev_pipeline import decide, MalformedAnswers  # noqa: E402


def mk_answers(
    scene_assessable=1.0,
    hp=0.0,
    htype="none",
    crosswalk=0.0,
    sig_present=0.0,
    sig_color="unclear",
):
    """Build a well-formed answers dict in the Decisions API shape."""
    return {
        "scene_assessable": {"type": "noul", "noul": scene_assessable},
        "hazard_proximity": {"type": "score", "score": hp},
        "hazard_type": {"type": "choice", "choice": htype},
        "crosswalk_present": {"type": "noul", "noul": crosswalk},
        "signal_box_present": {"type": "noul", "noul": sig_present},
        "pedestrian_signal_color": {"type": "choice", "choice": sig_color},
    }


class TestEscalation(unittest.TestCase):
    def test_unassessable_scene_escalates(self):
        a = mk_answers(scene_assessable=0.3, hp=1.5, htype="moving_vehicle")
        self.assertEqual(decide(a), "ESCALATE_UNREADABLE")

    def test_missing_field_fails_safe_not_open(self):
        """A malformed/incomplete answers dict must escalate, never
        silently resolve to WALK from partial data."""
        a = mk_answers()
        del a["hazard_proximity"]
        self.assertEqual(decide(a), "ESCALATE_UNREADABLE")

    def test_wrong_shape_fails_safe(self):
        a = mk_answers()
        a["hazard_proximity"] = "not a dict"
        self.assertEqual(decide(a), "ESCALATE_UNREADABLE")

    def test_empty_answers_dict_fails_safe(self):
        self.assertEqual(decide({}), "ESCALATE_UNREADABLE")


class TestMovingVehiclePriority(unittest.TestCase):
    """A moving vehicle in the path must BLOCK even at a red-lit
    crossing -- checked before the red-light WAIT (v1.2 reorder)."""

    def test_moving_vehicle_blocks_even_with_red_light(self):
        a = mk_answers(hp=1.2, htype="moving_vehicle", crosswalk=0.9, sig_present=0.9, sig_color="red")
        self.assertEqual(decide(a), "BLOCK")

    def test_moving_vehicle_below_threshold_does_not_block(self):
        a = mk_answers(hp=0.5, htype="moving_vehicle", crosswalk=0.9, sig_present=0.9, sig_color="red")
        # hp below T_STOP -- vehicle noted but not yet in the path -> falls to red check
        self.assertEqual(decide(a), "WAIT")


class TestRedSignal(unittest.TestCase):
    def test_red_at_marked_crosswalk_waits(self):
        a = mk_answers(hp=0.2, htype="none", crosswalk=0.9, sig_present=0.9, sig_color="red")
        self.assertEqual(decide(a), "WAIT")

    def test_red_at_rail_crossing_waits_even_without_crosswalk_markings(self):
        a = mk_answers(hp=0.8, htype="rail_or_track", crosswalk=0.0, sig_present=0.6, sig_color="red")
        self.assertEqual(decide(a), "WAIT")

    def test_distant_irrelevant_red_light_on_sidewalk_does_not_wait(self):
        """Regression test for the v1.0 bug: a red light visible in frame
        but not at a crossing relevant to this forward path must not
        force WAIT on someone just walking down a sidewalk."""
        a = mk_answers(hp=0.2, htype="none", crosswalk=0.1, sig_present=0.5, sig_color="red")
        self.assertEqual(decide(a), "WALK")


class TestCrosswalkYielding(unittest.TestCase):
    def test_crosswalk_green_walks(self):
        a = mk_answers(hp=0.3, htype="none", crosswalk=0.9, sig_present=0.9, sig_color="green")
        self.assertEqual(decide(a), "WALK")

    def test_crosswalk_no_signal_walks(self):
        a = mk_answers(hp=0.3, htype="none", crosswalk=0.9, sig_present=0.1, sig_color="unclear")
        self.assertEqual(decide(a), "WALK")

    def test_crosswalk_stopped_vehicle_still_walks(self):
        """A yielding/stopped vehicle at a crosswalk is not a hazard
        (Austrian StVO: cars must yield here)."""
        a = mk_answers(hp=0.9, htype="stopped_vehicle", crosswalk=0.9, sig_present=0.1, sig_color="unclear")
        self.assertEqual(decide(a), "WALK")

    def test_crosswalk_does_not_override_stairs(self):
        """A crosswalk does not make a real physical hazard safe."""
        a = mk_answers(hp=1.5, htype="stairs_or_drop", crosswalk=0.9, sig_present=0.1, sig_color="unclear")
        self.assertEqual(decide(a), "BLOCK")

    def test_crosswalk_does_not_override_rail(self):
        # sig_present=0.1 is below the 0.5 "signal present but unreadable"
        # gate, so this falls through to the physical hp>=T_STOP check.
        a = mk_answers(hp=1.3, htype="rail_or_track", crosswalk=0.9, sig_present=0.1, sig_color="unclear")
        self.assertEqual(decide(a), "BLOCK")
        # With sig_present>=0.5, the unreadable-signal-at-relevant-crossing
        # rule fires first instead.
        a2 = mk_answers(hp=1.3, htype="rail_or_track", crosswalk=0.9, sig_present=0.6, sig_color="unclear")
        self.assertEqual(decide(a2), "WAIT")


class TestUnreadableSignal(unittest.TestCase):
    def test_unreadable_signal_at_relevant_crossing_waits(self):
        a = mk_answers(hp=0.3, htype="none", crosswalk=0.9, sig_present=0.8, sig_color="unclear")
        self.assertEqual(decide(a), "WAIT")

    def test_unreadable_signal_not_relevant_crossing_walks(self):
        a = mk_answers(hp=0.1, htype="none", crosswalk=0.1, sig_present=0.1, sig_color="unclear")
        self.assertEqual(decide(a), "WALK")


class TestPhysicalHazardFallback(unittest.TestCase):
    def test_stairs_blocks_with_no_crosswalk_no_signal(self):
        a = mk_answers(hp=1.5, htype="stairs_or_drop", crosswalk=0.0, sig_present=0.0, sig_color="unclear")
        self.assertEqual(decide(a), "BLOCK")

    def test_clear_path_walks(self):
        a = mk_answers(hp=0.1, htype="none", crosswalk=0.0, sig_present=0.0, sig_color="unclear")
        self.assertEqual(decide(a), "WALK")

    def test_boundary_hp_exactly_at_threshold_blocks(self):
        a = mk_answers(hp=1.0, htype="other_obstacle", crosswalk=0.0, sig_present=0.0, sig_color="unclear")
        self.assertEqual(decide(a), "BLOCK")

    def test_boundary_hp_just_under_threshold_walks(self):
        a = mk_answers(hp=0.99, htype="other_obstacle", crosswalk=0.0, sig_present=0.0, sig_color="unclear")
        self.assertEqual(decide(a), "WALK")


if __name__ == "__main__":
    unittest.main()
