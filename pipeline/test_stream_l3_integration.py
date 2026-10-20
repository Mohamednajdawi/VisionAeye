"""
Integration tests for VisionAeyeStream.ask() -- proves L3 requests are
correctly routed through arbitration, with no network calls (L3's
model call is mocked; L1/L2/arbitration run for real since they're
already proven deterministic and free).
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
from stream import VisionAeyeStream  # noqa: E402
from l3_assistant import L3Error  # noqa: E402


class TestAskRoutesThroughArbitration(unittest.TestCase):
    def _make_stream(self):
        # prompt_path/api_key are irrelevant here since we never call
        # on_frame() or ask_about_image for real in these tests.
        stream = VisionAeyeStream.__new__(VisionAeyeStream)
        from arbitration import ArbitrationEngine
        stream.arbitration = ArbitrationEngine()
        stream.api_key = "fake-key"
        return stream

    def test_ask_speaks_immediately_when_no_hazard_active(self):
        stream = self._make_stream()
        with patch("stream.ask_about_image", return_value={"answer": "It's on the left shelf."}):
            result = stream.ask("fakeb64", "where's the oat milk?")
        kinds = [a.kind for a in result["audio_actions"]]
        self.assertEqual(kinds, ["SPEAK"])
        self.assertEqual(result["l3_result"]["answer"], "It's on the left shelf.")

    def test_ask_is_queued_not_spoken_during_active_hazard(self):
        """The critical safety case: a question asked while a hazard
        is active must NOT be spoken over it."""
        stream = self._make_stream()
        stream.arbitration.on_l1_decision("BLOCK")  # hazard active

        with patch("stream.ask_about_image", return_value={"answer": "It's on the left shelf."}):
            result = stream.ask("fakeb64", "where's the oat milk?")

        kinds = [a.kind for a in result["audio_actions"]]
        self.assertEqual(kinds, ["QUEUE"])

    def test_l3_transport_failure_speaks_honest_message_not_silence(self):
        """An L3Error must still produce audio (an honest failure
        message), never silence -- a silent failure on a question the
        user explicitly asked is worse than an honest 'try again'."""
        stream = self._make_stream()
        with patch("stream.ask_about_image", side_effect=L3Error("timeout")):
            result = stream.ask("fakeb64", "what's on the menu?")

        self.assertGreater(len(result["audio_actions"]), 0)
        self.assertIn("error", result["l3_result"])
        # The spoken text must be a real message, not empty.
        spoken = result["audio_actions"][0].text
        self.assertTrue(len(spoken) > 0)

    def test_queued_answer_resumes_after_hazard_clears(self):
        stream = self._make_stream()
        stream.arbitration.on_l1_decision("WAIT")  # hazard active

        with patch("stream.ask_about_image", return_value={"answer": "Milk is on aisle 3."}):
            stream.ask("fakeb64", "where's the milk?")

        actions = stream.arbitration.on_l1_decision("WALK")  # hazard clears
        kinds = [a.kind for a in actions]
        self.assertIn("RESUME", kinds)
        resume = next(a for a in actions if a.kind == "RESUME")
        self.assertEqual(resume.text, "Milk is on aisle 3.")


if __name__ == "__main__":
    unittest.main()
