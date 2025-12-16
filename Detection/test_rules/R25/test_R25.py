import ast
import unittest
import sys
import os
# Ensure the generated_rules_R25 module in the same directory is importable
sys.path.insert(0, os.path.dirname(__file__))
import generated_rules_R25  # The file generated for rule R25

class TestGeneratedRules25(unittest.TestCase):
    def setUp(self):
        # Capture reported messages
        self.messages = []
        def report(message):
            self.messages.append(message)
        # Monkey-patch the report function in the generated module
        generated_rules_R25.report = report

    def run_rule(self, code: str):
        """Parse code and run the R25 rule on its AST."""
        self.messages.clear()
        tree = ast.parse(code)
        # Add parent links if needed by predicates
        from generated_rules_R25 import add_parent_info
        add_parent_info(tree)
        # Execute the rule function
        generated_rules_R25.rule_R25(tree)

    # TRUE POSITIVES - Should detect missing temperature
    
    def test_openai_completion_without_temperature(self):
        """Should report when OpenAI completion is called without temperature."""
        code = """
import openai
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Hello world",
    max_tokens=100
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing temperature")
        self.assertIn("temperature", self.messages[0].lower())

    def test_openai_chat_completion_without_temperature(self):
        """Should report when OpenAI ChatCompletion is called without temperature."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing temperature in ChatCompletion")

    def test_langchain_llm_without_temperature(self):
        """Should report when LangChain LLM is initialized without temperature."""
        code = """
from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing temperature in LangChain")

    def test_anthropic_completion_without_temperature(self):
        """Should report when Anthropic completion is called without temperature."""
        code = """
import anthropic
client = anthropic.Anthropic()
response = client.completions.create(
    model="claude-2",
    prompt="Hello world",
    max_tokens_to_sample=100
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing temperature in Anthropic")

    # TRUE NEGATIVES - Should NOT report when temperature is set

    def test_openai_completion_with_temperature(self):
        """Should not report when temperature is explicitly set."""
        code = """
import openai
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Hello world",
    temperature=0.7,
    max_tokens=100
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when temperature is provided")

    def test_openai_chat_with_zero_temperature(self):
        """Should not report when temperature is set to 0 (deterministic)."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when temperature=0")

    def test_langchain_with_temperature(self):
        """Should not report when LangChain LLM has temperature set."""
        code = """
from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when LangChain has temperature")

    def test_temperature_as_variable(self):
        """Should not report when temperature is provided via variable."""
        code = """
import openai
temp = 0.8
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Hello",
    temperature=temp
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when temperature is from variable")

    # FALSE POSITIVES - Edge cases that might trigger incorrectly

    def test_non_llm_api_call(self):
        """Should not report for non-LLM API calls that don't need temperature."""
        code = """
import requests
response = requests.get("https://api.example.com/data")
data = response.json()
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report for non-LLM API calls")

    def test_regular_function_with_similar_name(self):
        """Should not report for regular functions with similar names."""
        code = """
def create_completion(text, max_length=100):
    return text[:max_length]
    
result = create_completion("Hello world")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should not report for user-defined functions")

    # FALSE NEGATIVES - Cases that might be missed

    def test_llm_call_via_client_object(self):
        """Should detect missing temperature even with client pattern."""
        code = """
from openai import OpenAI
client = OpenAI()
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Say hello"
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Should detect missing temperature with client pattern")

    def test_kwargs_without_temperature(self):
        """Should detect when kwargs are used but temperature is missing."""
        code = """
import openai
params = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 50
}
response = openai.ChatCompletion.create(**params)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Should detect missing temperature in kwargs")

    def test_huggingface_pipeline_without_temperature(self):
        """Should detect HuggingFace pipeline without temperature."""
        code = """
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello", max_length=50)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Should detect missing temperature in HF pipeline")

    # COMPLEX CASES

    def test_multiple_llm_calls_mixed(self):
        """Should only report for calls missing temperature."""
        code = """
import openai

# This should be reported
response1 = openai.Completion.create(
    model="text-davinci-003",
    prompt="First prompt"
)

# This should NOT be reported
response2 = openai.Completion.create(
    model="text-davinci-003", 
    prompt="Second prompt",
    temperature=0.5
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 1, "Should report only one missing temperature")

    def test_temperature_in_config_object(self):
        """Should not report when temperature is in a configuration object."""
        code = """
import openai
config = {"temperature": 0.7, "max_tokens": 100}
response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt="Hello",
    **config
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Should recognize temperature in config")

    def test_openai_completion_without_temperature_but_with_top_p(self):
        """Should not report when top_p is set even if temperature is omitted."""
        code = """
import openai
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Hello world",
    top_p=0.9,
    max_tokens=100
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when top_p is provided")

    def test_openai_chat_completion_without_temperature_but_with_top_p(self):
        """Should not report when top_p is set even if temperature is omitted (chat)."""
        code = """
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    top_p=0.8
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when top_p is provided in ChatCompletion")

    def test_anthropic_completion_without_temperature_but_with_top_p(self):
        """Should not report when top_p is set even if temperature is omitted (Anthropic)."""
        code = """
import anthropic
client = anthropic.Anthropic()
response = client.completions.create(
    model="claude-2",
    prompt="Hello world",
    top_p=0.9,
    max_tokens_to_sample=100
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when top_p is provided in Anthropic")

    def test_anthropic_completion_without_temperature_but_with_top_k(self):
        """Should not report when top_k is set even if temperature is omitted (Anthropic)."""
        code = """
import anthropic
client = anthropic.Anthropic()
response = client.completions.create(
    model="claude-2",
    prompt="Hello world",
    top_k=40,
    max_tokens_to_sample=100
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when top_k is provided in Anthropic")

    def test_kwargs_with_top_p_no_temperature_should_not_report(self):
        """Should not report when kwargs include top_p even if temperature is missing."""
        code = """
import openai
params = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}],
    "top_p": 0.95,
    "max_tokens": 50
}
response = openai.ChatCompletion.create(**params)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when kwargs include top_p")

    def test_huggingface_pipeline_without_temperature_but_with_top_k(self):
        """Should not report HF generation when top_k is set even if temperature is omitted."""
        code = """
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello", max_length=50, top_k=40)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when HF call provides top_k")

    def test_huggingface_pipeline_without_temperature_but_with_top_p(self):
        """Should not report HF generation when top_p is set even if temperature is omitted."""
        code = """
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello", max_length=50, top_p=0.9)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect a report when HF call provides top_p")


if __name__ == '__main__':
    unittest.main()