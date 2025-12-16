import ast
import unittest
import sys
import os

# Ensure the generated_rules_R26 module in the same directory is importable
sys.path.insert(0, os.path.dirname(__file__))
import generated_rules_R26  # The file generated for rule R26 (No Model Version Pinning)


class TestGeneratedRules26(unittest.TestCase):
    def setUp(self):
        # Capture reported messages
        self.messages = []

        def report(message):
            self.messages.append(message)

        # Monkey-patch the report function in the generated module
        generated_rules_R26.report = report

    def run_rule(self, code: str):
        """Parse code and run the R26 rule on its AST."""
        self.messages.clear()
        tree = ast.parse(code)
        # Add parent links if needed by predicates
        from generated_rules_R26 import add_parent_info
        add_parent_info(tree)
        # Execute the rule function
        generated_rules_R26.rule_R26(tree)

    # ========== TRUE POSITIVES (should report UNPINNED) ==========

    def test_openai_chat_unpinned_alias(self):
        """OpenAI alias (no date) should be reported."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for unpinned model (OpenAI alias)")
        self.assertIn("version", self.messages[0].lower())

    def test_openai_completion_unpinned_alias(self):
        """OpenAI Completion alias should be reported."""
        code = """
import openai
response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt="Hello"
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for unpinned model (OpenAI alias)")

    def test_anthropic_latest_unpinned(self):
        """Anthropic 'latest' alias should be reported."""
        code = """
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-3-5-haiku-latest",
    messages=[{"role":"user","content":"hi"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for Anthropic latest alias (unpinned)")

    def test_gemini_unpinned_alias(self):
        """Gemini alias w/o explicit revision should be reported."""
        code = """
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-pro")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for unpinned Gemini alias")

    def test_hf_from_pretrained_without_revision(self):
        """HF from_pretrained without revision should be reported."""
        code = """
from transformers import AutoModel
m = AutoModel.from_pretrained("some-org/some-model")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report when revision is missing in from_pretrained")

    def test_hf_from_pretrained_revision_main(self):
        """HF from_pretrained with revision='main' should be reported (still unpinned)."""
        code = """
from transformers import AutoModel
m = AutoModel.from_pretrained("some-org/some-model", revision="main")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report when revision='main' (unpinned)")

    def test_ollama_latest_tag_unpinned(self):
        """Ollama with :latest should be reported."""
        code = """
import subprocess
subprocess.run(["ollama","run","llama3:latest"])
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for Ollama :latest tag")

    def test_kwargs_unpinned_model_in_params(self):
        """**params where model is alias should be reported."""
        code = """
import openai
params = {
    "model": "gpt-4o",
    "messages": [{"role":"user","content":"Hi"}]
}
openai.ChatCompletion.create(**params)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for unpinned model inside **params")

    def test_ollama_shell_string_latest_unpinned(self):
        """Ollama with shell string 'ollama run ...:latest' should be reported."""
        code = """
import subprocess
subprocess.run("ollama run llama3:latest", shell=True)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for Ollama latest in shell string")

    # ========== TRUE NEGATIVES (should NOT report when PINNED) ==========

    def test_openai_chat_pinned_dated(self):
        """OpenAI with dated ID should NOT be reported."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4o-2024-11-20",
    messages=[{"role": "user", "content": "Hello"}]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report for pinned OpenAI model")

    def test_openai_turbo_pinned_dated(self):
        """OpenAI turbo with date should NOT be reported."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4-turbo-2024-04-09",
    messages=[{"role": "user", "content": "Hello"}]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report for dated turbo model")

    def test_anthropic_snapshot_pinned(self):
        """Anthropic with date snapshot should NOT be reported."""
        code = """
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-3-5-haiku-20241022",
    messages=[{"role":"user","content":"hi"}]
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report for Anthropic snapshot date")

    def test_gemini_pinned_revision(self):
        """Gemini with explicit revision (e.g., -002) should NOT be reported."""
        code = """
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-pro-002")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report for pinned Gemini revision")

    def test_hf_from_pretrained_with_commit(self):
        """HF from_pretrained pinned via commit/tag should NOT be reported."""
        code = """
from transformers import AutoModel
m = AutoModel.from_pretrained("some-org/some-model", revision="2a1f6a7")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when revision is a commit/tag")

    def test_ollama_specific_tag(self):
        """Ollama with specific tag (non-latest) should NOT be reported."""
        code = """
import subprocess
subprocess.run(["ollama","run","llama3:8b-instruct"])
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report for specific Ollama tag")

    def test_kwargs_pinned_model_in_params(self):
        """**params with pinned model should NOT be reported."""
        code = """
import openai
params = {
    "model": "gpt-4o-2024-11-20",
    "messages": [{"role":"user","content":"Hi"}]
}
openai.ChatCompletion.create(**params)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report for pinned model inside **params")

    # ========== FALSE POSITIVES (should NOT report) ==========

    def test_non_llm_api_call(self):
        """Non-LLM API should NOT be reported."""
        code = """
import requests
r = requests.get("https://api.example.com/data")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report for non-LLM API calls")

    def test_user_defined_from_pretrained(self):
        """User function named from_pretrained should NOT be reported."""
        code = """
def from_pretrained(name, revision=None):
    return f"custom:{name}:{revision}"

m = from_pretrained("foo", revision="main")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report for user-defined functions")

    # ========== COMPLEX CASES ==========

    def test_multiple_calls_mixed(self):
        """Only unpinned calls should trigger a report."""
        code = """
import openai
# Unpinned
resp1 = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role":"user","content":"A"}])
# Pinned
resp2 = openai.ChatCompletion.create(model="gpt-4o-2024-11-20", messages=[{"role":"user","content":"B"}])
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 1, "Should report only the unpinned call")

    def test_mixed_hf_and_ollama(self):
        """HF without revision and Ollama :latest should both report (2 reports)."""
        code = """
from transformers import AutoModel
import subprocess

m1 = AutoModel.from_pretrained("some-org/some-model")  # missing revision
subprocess.run(["ollama","run","llama3:latest"])       # latest
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 2, "Should report HF missing revision and Ollama latest")

    def test_hf_with_tag_and_ollama_specific(self):
        """HF with tag/commit and Ollama specific tag should NOT report."""
        code = """
from transformers import AutoModel
import subprocess

m1 = AutoModel.from_pretrained("some-org/some-model", revision="v2.0.1")
subprocess.run(["ollama","run","llama3:q4_K_M"])
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report when both are pinned/specific")


if __name__ == '__main__':
    unittest.main()
