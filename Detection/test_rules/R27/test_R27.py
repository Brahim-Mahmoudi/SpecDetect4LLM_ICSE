import ast
import unittest
import sys
import os

# Ensure the generated_rules_R27 module in the same directory is importable
sys.path.insert(0, os.path.dirname(__file__))
import generated_rules_R27  # The file generated for rule R27 (No System Message)


class TestGeneratedRules27(unittest.TestCase):
    def setUp(self):
        # Capture reported messages
        self.messages = []

        def report(message):
            self.messages.append(message)

        # Monkey-patch the report function in the generated module
        generated_rules_R27.report = report

    def run_rule(self, code: str):
        """Parse code and run the R27 rule on its AST."""
        self.messages.clear()
        tree = ast.parse(code)
        # Add parent links if needed by predicates
        from generated_rules_R27 import add_parent_info
        add_parent_info(tree)
        # Execute the rule function
        generated_rules_R27.rule_R27(tree)

    # ========== TRUE POSITIVES (should REPORT: missing system/instructions) ==========

    def test_openai_chat_missing_system(self):
        """OpenAI ChatCompletion without a system message should be reported."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system message (OpenAI chat)")
        self.assertIn("system", self.messages[0].lower())

    def test_openai_responses_missing_instructions(self):
        """OpenAI Responses API without 'instructions' (and chat-like input) should be reported."""
        code = """
from openai import OpenAI
client = OpenAI()
resp = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role":"user","content":"Hi there"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report when 'instructions' are missing (OpenAI Responses)")

    def test_anthropic_missing_system(self):
        """Anthropic messages.create without 'system' should be reported."""
        code = """
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-3-5-haiku-20241022",
    messages=[{"role":"user","content":"hi"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system (Anthropic)")

    def test_kwargs_messages_without_system(self):
        """**params with messages lacking a system role should be reported."""
        code = """
import openai
params = {
    "model": "gpt-4o",
    "messages": [{"role":"user","content":"Only user"}]
}
openai.ChatCompletion.create(**params)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system inside **params")

    # ========== TRUE NEGATIVES (should NOT REPORT when system/instructions present) ==========

    def test_openai_chat_with_system(self):
        """OpenAI ChatCompletion with system message should NOT be reported."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Hello"}
    ]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when system message is present")

    def test_openai_responses_with_instructions(self):
        """OpenAI Responses API with 'instructions' should NOT be reported."""
        code = """
from openai import OpenAI
client = OpenAI()
resp = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are precise and concise.",
    input="Summarize: ...")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when 'instructions' is provided")

    def test_anthropic_with_system(self):
        """Anthropic messages.create with 'system' should NOT be reported."""
        code = """
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-3-5-haiku-20241022",
    system="You are a domain expert.",
    messages=[{"role":"user","content":"hi"}]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when Anthropic 'system' is set")

    # ========== FALSE POSITIVES (APIs where system is not applicable) ==========

    def test_openai_legacy_completion(self):
        """Legacy (non-chat) Completion API should NOT be reported."""
        code = """
import openai
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a haiku"
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report for non-chat completion endpoint")

    def test_non_llm_api_call(self):
        """Non-LLM API should NOT be reported."""
        code = """
import requests
r = requests.get("https://api.example.com/data")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report for non-LLM API calls")

    # ========== COMPLEX CASES ==========

    def test_multiple_calls_mixed(self):
        """Only the call without a system message should trigger a report."""
        code = """
import openai
# Missing system -> should report
resp1 = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"A"}]
)
# With system -> should not report
resp2 = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"system","content":"Be terse."},
              {"role":"user","content":"B"}]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 1, "Should report only the missing-system call")

    def test_responses_mixed_with_and_without_instructions(self):
        """Responses API: one without instructions (report) and one with (no report)."""
        code = """
from openai import OpenAI
client = OpenAI()

# No instructions -> report
a = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role":"user","content":"Hi"}]
)

# With instructions -> no report
b = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are structured.",
    input=[{"role":"user","content":"Hello"}]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 1, "Should report exactly one missing-instructions call")

    # ===================== GEMINI (google.generativeai) =====================

    def test_gemini_no_system_instruction_generate_content(self):
        """GenAI: GenerativeModel(...) sans system_instruction + generate_content -> REPORT."""
        code = """
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-pro")
resp = model.generate_content("Hello")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system_instruction (generate_content)")

    def test_gemini_with_system_instruction_generate_content(self):
        """GenAI: GenerativeModel(..., system_instruction=...) -> NO REPORT."""
        code = """
import google.generativeai as genai
model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                              system_instruction="You are a terse expert.")
resp = model.generate_content("Hello")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when system_instruction is provided")

    def test_gemini_start_chat_missing_system(self):
        """GenAI: start_chat avec history mais sans system_instruction -> REPORT."""
        code = """
import google.generativeai as genai
m = genai.GenerativeModel("gemini-1.5-pro")
chat = m.start_chat(history=[{"role":"user","parts":["hi"]}])
ans = chat.send_message("ok?")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system_instruction (start_chat)")

    def test_gemini_start_chat_with_system(self):
        """GenAI: start_chat avec system_instruction au constructeur -> NO REPORT."""
        code = """
import google.generativeai as genai
m = genai.GenerativeModel("gemini-1.5-pro", system_instruction="Be concise.")
chat = m.start_chat(history=[{"role":"user","parts":["hi"]}])
ans = chat.send_message("ok?")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when system_instruction is present")

    def test_gemini_kwargs_model_params_missing_system(self):
        """GenAI: kwargs littéral sans system_instruction -> REPORT."""
        code = """
import google.generativeai as genai
params = {"model_name": "gemini-1.5-pro"}  # no system_instruction
m = genai.GenerativeModel(**params)
m.generate_content("test")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system_instruction in **params")

    def test_gemini_empty_system_instruction_optional_subsmell(self):
        """Optionnel: considérer system_instruction="" comme 'Empty System Message' (si la règle le gère)."""
        code = """
import google.generativeai as genai
m = genai.GenerativeModel("gemini-1.5-pro", system_instruction="")
m.generate_content("test")
"""
        self.run_rule(code)
        # Si R27 ne gère pas 'empty', on peut soit attendre un report, soit ignorer ce test.
        # Ici on attend un report si tu cibles aussi le sous-cas 'Empty System Message'.
        self.assertTrue(self.messages, "Expected a report for empty system_instruction")

    # ===================== GEMINI (Vertex AI SDK) =====================

    def test_vertexai_no_system_instruction(self):
        """VertexAI: GenerativeModel(...) sans system_instruction -> REPORT."""
        code = """
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-pro")
out = model.generate_content("Hello")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing system_instruction (Vertex AI)")

    def test_vertexai_with_system_instruction(self):
        """VertexAI: GenerativeModel(..., system_instruction=...) -> NO REPORT."""
        code = """
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-pro", system_instruction="You answer in JSON.")
out = model.generate_content("Hello")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when system_instruction is set (Vertex AI)")


if __name__ == '__main__':
    unittest.main()
