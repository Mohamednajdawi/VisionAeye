"""
Unit tests for the arbitration engine. Pure logic, no models, no I/O --
the whole point is these can be written and trusted before any real
L2/L3 model exists.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from arbitration import ArbitrationEngine, AudioAction, L3State  # noqa: E402


class TestSteadyStateWalk(unittest.TestCase):
    def test_repeated_walk_emits_nothing(self):
        e = ArbitrationEngine()
        self.assertEqual(e.on_l1_decision("WALK"), [])
        self.assertEqual(e.on_l1_decision("WALK"), [])
        self.assertEqual(e.l3_state, L3State.IDLE)

    def test_walk_with_l3_idle_is_silent(self):
        e = ArbitrationEngine()
        actions = e.on_l1_decision("WALK")
        self.assertEqual(actions, [])


class TestL3NormalFlow(unittest.TestCase):
    def test_l3_request_speaks_when_no_hazard(self):
        e = ArbitrationEngine()
        actions = e.on_l3_request("The soup of the day is tomato.")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "SPEAK")
        self.assertEqual(e.l3_state, L3State.SPEAKING)

    def test_l3_done_returns_to_idle(self):
        e = ArbitrationEngine()
        e.on_l3_request("hello")
        e.on_l3_done()
        self.assertEqual(e.l3_state, L3State.IDLE)


class TestHazardInterruptsL3(unittest.TestCase):
    """The core safety contract: L1 always wins, every time, no
    exceptions, regardless of what L3 is doing."""

    def test_wait_interrupts_speaking_l3(self):
        e = ArbitrationEngine()
        e.on_l3_request("The pasta special today is...")
        self.assertEqual(e.l3_state, L3State.SPEAKING)

        actions = e.on_l1_decision("WAIT")
        kinds = [a.kind for a in actions]
        self.assertIn("INTERRUPT", kinds)
        self.assertIn("SPEAK_URGENT", kinds)
        self.assertEqual(e.l3_state, L3State.PAUSED)

    def test_block_interrupts_speaking_l3(self):
        e = ArbitrationEngine()
        e.on_l3_request("Aisle 3 has the cereal you're looking for.")
        actions = e.on_l1_decision("BLOCK")
        self.assertIn("INTERRUPT", [a.kind for a in actions])
        self.assertEqual(e.l3_state, L3State.PAUSED)

    def test_escalate_unreadable_interrupts_l3(self):
        """Escalations are exactly as urgent as BLOCK -- "I can't tell
        if this is safe" must interrupt just like "this is unsafe"."""
        e = ArbitrationEngine()
        e.on_l3_request("...")
        actions = e.on_l1_decision("ESCALATE_UNREADABLE")
        self.assertIn("INTERRUPT", [a.kind for a in actions])

    def test_escalate_uncertain_interrupts_l3(self):
        e = ArbitrationEngine()
        e.on_l3_request("...")
        actions = e.on_l1_decision("ESCALATE_UNCERTAIN")
        self.assertIn("INTERRUPT", [a.kind for a in actions])

    def test_hazard_with_idle_l3_speaks_urgent_without_interrupt(self):
        """No need to emit an INTERRUPT action if L3 wasn't speaking --
        nothing to interrupt."""
        e = ArbitrationEngine()
        actions = e.on_l1_decision("WAIT")
        kinds = [a.kind for a in actions]
        self.assertNotIn("INTERRUPT", kinds)
        self.assertIn("SPEAK_URGENT", kinds)


class TestResumeAfterHazardClears(unittest.TestCase):
    def test_resume_paused_l3_after_hazard_clears(self):
        e = ArbitrationEngine()
        e.on_l3_request("The soup of the day is tomato basil, served with")
        e.on_l1_decision("WAIT")  # interrupts
        self.assertEqual(e.l3_state, L3State.PAUSED)

        actions = e.on_l1_decision("WALK")  # hazard clears
        self.assertEqual(e.l3_state, L3State.SPEAKING)
        kinds = [a.kind for a in actions]
        self.assertIn("RESUME", kinds)

    def test_no_resume_action_while_hazard_still_active(self):
        e = ArbitrationEngine()
        e.on_l3_request("hello")
        e.on_l1_decision("WAIT")
        actions = e.on_l1_decision("WAIT")  # still WAIT, not cleared
        self.assertNotIn("RESUME", [a.kind for a in actions])


class TestOngoingHazardReannouncement(unittest.TestCase):
    """Silence during an ongoing hazard is unsafe -- the user must hear
    the current state, not assume yesterday's announcement still holds."""

    def test_wait_then_block_reannounces_with_new_severity(self):
        e = ArbitrationEngine()
        e.on_l1_decision("WAIT")
        actions = e.on_l1_decision("BLOCK")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "SPEAK_URGENT")
        self.assertEqual(actions[0].text, "BLOCK")

    def test_repeated_identical_urgent_decision_still_reannounces(self):
        """Deliberate choice: repeating BLOCK every tick while a hazard
        persists is safer than going silent and letting the user assume
        it's cleared. (If this becomes annoying in practice, debounce
        at the L2 layer -- not here, where safety must win over UX.)"""
        e = ArbitrationEngine()
        e.on_l1_decision("BLOCK")
        actions = e.on_l1_decision("BLOCK")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "SPEAK_URGENT")


class TestL3QueuedDuringHazard(unittest.TestCase):
    def test_l3_request_during_active_hazard_is_queued_not_spoken(self):
        e = ArbitrationEngine()
        e.on_l1_decision("WAIT")  # hazard active, L3 was idle
        actions = e.on_l3_request("Here's what's on the menu...")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "QUEUE")
        self.assertEqual(e.l3_state, L3State.IDLE)  # not speaking

    def test_queued_l3_request_resumes_after_hazard_clears(self):
        e = ArbitrationEngine()
        e.on_l1_decision("WAIT")
        e.on_l3_request("Here's what's on the menu...")
        actions = e.on_l1_decision("WALK")
        kinds = [a.kind for a in actions]
        self.assertIn("RESUME", kinds)
        resume_action = next(a for a in actions if a.kind == "RESUME")
        self.assertEqual(resume_action.text, "Here's what's on the menu...")


class TestMultiCycleIntegration(unittest.TestCase):
    """A fuller scenario: L3 reading a long menu description, hazard
    fires mid-sentence, clears, L3 resumes, finishes normally."""

    def test_full_cycle(self):
        e = ArbitrationEngine()

        # L3 starts describing the menu.
        actions = e.on_l3_request("Today's specials are: tomato soup,")
        self.assertEqual(actions[0].kind, "SPEAK")

        # A car appears -- must interrupt immediately.
        actions = e.on_l1_decision("BLOCK")
        self.assertEqual([a.kind for a in actions], ["INTERRUPT", "SPEAK_URGENT"])
        self.assertEqual(e.l3_state, L3State.PAUSED)

        # Hazard persists one more tick -- re-announce, no new INTERRUPT needed.
        actions = e.on_l1_decision("BLOCK")
        self.assertEqual([a.kind for a in actions], ["SPEAK_URGENT"])

        # Car passes, path clear again -- resume the menu.
        actions = e.on_l1_decision("WALK")
        self.assertEqual([a.kind for a in actions], ["RESUME"])
        self.assertEqual(e.l3_state, L3State.SPEAKING)

        # L3 finishes normally.
        e.on_l3_done()
        self.assertEqual(e.l3_state, L3State.IDLE)


if __name__ == "__main__":
    unittest.main()
