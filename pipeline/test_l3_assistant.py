"""
Unit tests for pipeline/l3_assistant.py. Mocks urllib so these run
without an API key or network access -- fast, deterministic, same
philosophy as test_pediev_pipeline.py / test_arbitration.py /
test_l2_buffer.py.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from l3_assistant import ask_about_image, L3Error  # noqa: E402


def _mock_response(body: dict, status: int = 200):
    """Build a context-manager-compatible fake for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body).encode("utf-8")
    mock_resp.status = status
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = False
    return mock_cm


class TestAskAboutImage(unittest.TestCase):
    def test_successful_call_returns_answer(self):
        body = {
            "model": "openai/gpt-4o-mini",
            "choices": [{"message": {"content": "The soup of the day is tomato basil."}}],
            "usage": {"total_cost": 0.0012},
        }
        with patch("l3_assistant.urllib.request.urlopen", return_value=_mock_response(body)):
            result = ask_about_image("fakeb64", "what's the soup?", api_key="fake-key")
        self.assertEqual(result["answer"], "The soup of the day is tomato basil.")
        self.assertEqual(result["cost"], 0.0012)
        self.assertEqual(result["model"], "openai/gpt-4o-mini")
        self.assertIsInstance(result["lat"], float)

    def test_answer_is_stripped_of_whitespace(self):
        body = {"choices": [{"message": {"content": "  trimmed answer \n"}}]}
        with patch("l3_assistant.urllib.request.urlopen", return_value=_mock_response(body)):
            result = ask_about_image("fakeb64", "q", api_key="fake-key")
        self.assertEqual(result["answer"], "trimmed answer")

    def test_missing_cost_field_does_not_crash(self):
        """OpenRouter's total_cost field is not guaranteed present on
        every provider -- must degrade gracefully, not raise."""
        body = {"choices": [{"message": {"content": "answer"}}], "usage": {}}
        with patch("l3_assistant.urllib.request.urlopen", return_value=_mock_response(body)):
            result = ask_about_image("fakeb64", "q", api_key="fake-key")
        self.assertIsNone(result["cost"])

    def test_malformed_response_raises_l3error_not_keyerror(self):
        """A malformed API response must raise our own exception type,
        never leak a raw KeyError/IndexError to the caller -- callers
        depend on catching L3Error specifically to speak an honest
        failure message."""
        body = {"unexpected": "shape"}
        with patch("l3_assistant.urllib.request.urlopen", return_value=_mock_response(body)):
            with self.assertRaises(L3Error):
                ask_about_image("fakeb64", "q", api_key="fake-key")

    def test_http_error_raises_l3error(self):
        import urllib.error
        fake_error = urllib.error.HTTPError(
            url="http://x", code=429, msg="rate limited",
            hdrs=None, fp=MagicMock(read=lambda: b'{"error":"rate limited"}'),
        )
        with patch("l3_assistant.urllib.request.urlopen", side_effect=fake_error):
            with self.assertRaises(L3Error):
                ask_about_image("fakeb64", "q", api_key="fake-key")

    def test_network_error_raises_l3error(self):
        import urllib.error
        fake_error = urllib.error.URLError("connection refused")
        with patch("l3_assistant.urllib.request.urlopen", side_effect=fake_error):
            with self.assertRaises(L3Error):
                ask_about_image("fakeb64", "q", api_key="fake-key")


if __name__ == "__main__":
    unittest.main()
