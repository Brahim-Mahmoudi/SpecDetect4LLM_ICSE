import ast
import unittest
import sys
import os

# Ensure the generated_rules_R28 module in the same directory is importable
sys.path.insert(0, os.path.dirname(__file__))
import generated_rules_R28  # The file generated for rule R28 (Unbounded Max Metrics)


class TestGeneratedRules28(unittest.TestCase):
    def setUp(self):
        # Capture reported messages
        self.messages = []

        def report(message):
            self.messages.append(message)

        # Monkey-patch the report function in the generated module
        generated_rules_R28.report = report

    def run_rule(self, code: str):
        """Parse code and run the R28 rule on its AST."""
        self.messages.clear()
        tree = ast.parse(code)
        from generated_rules_R28 import add_parent_info
        add_parent_info(tree)
        generated_rules_R28.rule_R28(tree)

    # ========== TRUE POSITIVES (should report UNBOUNDED) ==========

    def test_openai_responses_no_max_tokens(self):
        """OpenAI Responses API without max_output_tokens -> REPORT."""
        code = """
from openai import OpenAI
client = OpenAI()
resp = client.responses.create(
    model="gpt-4.1-mini",
    input="Hello world"
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing max_output_tokens")

    def test_chatcompletion_no_timeout(self):
        """ChatCompletion without timeout -> REPORT."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"Hello"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing timeout/max_tokens")

    def test_anthropic_no_max_tokens(self):
        """Anthropic call without max_tokens -> REPORT."""
        code = """
import anthropic
c = anthropic.Anthropic()
resp = c.messages.create(
    model="claude-3-5-haiku",
    messages=[{"role":"user","content":"hi"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing max_tokens")

    def test_gemini_no_bounds(self):
        """Gemini generate_content without system_instruction or max_output_tokens -> REPORT."""
        code = """
import google.generativeai as genai
m = genai.GenerativeModel("gemini-1.5-pro")
resp = m.generate_content("hello")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for unbounded Gemini call")

    # ========== TRUE NEGATIVES (bounded calls should NOT report) ==========

    def test_openai_responses_with_max_tokens_and_timeout(self):
        """OpenAI Responses with max_output_tokens and timeout -> NO REPORT."""
        code = """
from openai import OpenAI
client = OpenAI()
with client.with_options(timeout=30):
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input="Hello world",
        max_output_tokens=128
    )
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when bounded")

    def test_anthropic_with_max_tokens(self):
        """Anthropic with max_tokens should NOT report."""
        code = """
import anthropic
c = anthropic.Anthropic()
resp = c.messages.create(
    model="claude-3-5-haiku",
    messages=[{"role":"user","content":"hi"}],
    max_tokens=256
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when max_tokens present")

    def test_gemini_with_limits(self):
        """Gemini with system_instruction + bounded output -> NO REPORT."""
        code = """
import google.generativeai as genai
m = genai.GenerativeModel("gemini-1.5-pro", system_instruction="Be concise.")
resp = m.generate_content("hello", generation_config={"max_output_tokens": 128})
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when Gemini has max_output_tokens")

    # ========== COMPLEX CASES ==========

    def test_mixed_calls(self):
        """One bounded, one unbounded -> should report only the unbounded one."""
        code = """
import openai
# Unbounded
resp1 = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"A"}]
)

# Bounded
resp2 = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"B"}],
    max_tokens=200
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 1, "Should report only the unbounded call")


if __name__ == '__main__':
    unittest.main()
