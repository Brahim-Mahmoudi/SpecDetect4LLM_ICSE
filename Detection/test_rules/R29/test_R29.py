import ast
import unittest
import sys
import os

# Ensure the generated_rules_R29 module in the same directory is importable
sys.path.insert(0, os.path.dirname(__file__))
import generated_rules_R29  # The file generated for rule R29 (No Structured Output in Pipeline)


class TestGeneratedRules29(unittest.TestCase):
    def setUp(self):
        # Capture reported messages
        self.messages = []

        def report(message):
            self.messages.append(message)

        # Monkey-patch the report function in the generated module
        generated_rules_R29.report = report

    def run_rule(self, code: str):
        """Parse code and run the R29 rule on its AST."""
        self.messages.clear()
        tree = ast.parse(code)
        # make sure parent pointers exist
        if hasattr(generated_rules_R29, "add_parent_info"):
            generated_rules_R29.add_parent_info(tree)
        generated_rules_R29.rule_R29(tree)

    # ========== TRUE POSITIVES (should report NO STRUCTURED OUTPUT) ==========

    def test_langchain_llmchain_unstructured(self):
        """LangChain LLMChain without structured output -> REPORT."""
        code = """
from langchain.chains import LLMChain
from langchain.llms import OpenAI
llm = OpenAI()
chain = LLMChain(llm=llm, prompt="Tell me something")
resp = chain.run("hello")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for unstructured LangChain pipeline")

    def test_openai_json_missing(self):
        """OpenAI call in pipeline without response_format=json -> REPORT."""
        code = """
import openai
resp = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"Give me key/value pairs"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for missing response_format=json")

    def test_openai_responses_unstructured(self):
        """OpenAI Responses API without response_format/parser -> REPORT."""
        code = """
from openai import OpenAI
client = OpenAI()
res = client.responses.create(
    model="gpt-4o",
    input=[{"role":"user","content":"Return fields a,b"}]
)
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected a report for responses.create without structured output")

    def test_transformers_pipeline_text_generation_var_call_unstructured(self):
        """transformers.pipeline('text-generation') then var(...) -> REPORT."""
        code = """
from transformers import pipeline
gen = pipeline("text-generation")
out = gen("hello")
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected report for transformers text-generation without structured output")

    def test_langchain_invoke_unstructured(self):
        """LangChain chain.invoke without parser -> REPORT."""
        code = """
from langchain.chains import LLMChain
from langchain.llms import OpenAI
llm = OpenAI()
chain = LLMChain(llm=llm, prompt="Tell me something")
out = chain.invoke({"input":"hey"})
"""
        self.run_rule(code)
        self.assertTrue(self.messages, "Expected report for chain.invoke without parser")

    def test_multiple_unstructured_calls(self):
        """Two unstructured calls -> expect two reports."""
        code = """
import openai
r1 = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role":"user","content":"A"}])
r2 = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role":"user","content":"B"}])
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 2, "Expected exactly two reports for two unstructured calls")

    # ========== TRUE NEGATIVES (structured output should NOT report) ==========

    def test_openai_with_json_format(self):
        """OpenAI with response_format=json -> NO REPORT."""
        code = """
import openai
resp = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"Give me key/value pairs"}],
    response_format="json"
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when response_format=json present")

    def test_openai_with_json_object_format(self):
        """OpenAI with response_format=json_object -> NO REPORT."""
        code = """
import openai
resp = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"Give me key/value pairs"}],
    response_format="json_object"
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when response_format=json_object present")

    def test_openai_responses_with_json_format(self):
        """OpenAI Responses API with response_format=json -> NO REPORT."""
        code = """
from openai import OpenAI
client = OpenAI()
res = client.responses.create(
    model="gpt-4o",
    input=[{"role":"user","content":"Return fields a,b"}],
    response_format="json"
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when responses.create has response_format=json")

    def test_langchain_with_parser(self):
        """LangChain with StructuredOutputParser -> NO REPORT."""
        code = """
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import LLMChain
from langchain.llms import OpenAI

parser = StructuredOutputParser.from_response_schemas([])
llm = OpenAI()
chain = LLMChain(llm=llm, prompt="Tell me something", output_parser=parser)
resp = chain.run("hello")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when StructuredOutputParser is used")

    def test_langchain_predict_with_parser(self):
        """LLMChain(..., output_parser=...) then predict(...) -> NO REPORT."""
        code = """
from langchain.output_parsers import StructuredOutputParser
from langchain.chains import LLMChain
from langchain.llms import OpenAI

parser = StructuredOutputParser.from_response_schemas([])
llm = OpenAI()
chain = LLMChain(llm=llm, prompt="Tell me something", output_parser=parser)
res = chain.predict(input="hi")
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Did not expect report when predict() has parser set on chain")

    # ========== FALSE POSITIVES TO AVOID ==========

    def test_mcp_get_tools_should_not_report(self):
        """MCP control-plane call (no LLM generation) -> NO REPORT."""
        code = """
class MultiServerMCPClient:
    def __init__(self, servers): pass
    async def get_tools(self): pass

import asyncio
def validate_mcp_config(servers):
    async def validate():
        client = MultiServerMCPClient(servers)
        await client.get_tools()
    try:
        asyncio.run(validate())
    except Exception as e:
        pass
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "MCP get_tools() is not LLM generation; should not report")

    def test_non_llm_object_with_run_method(self):
        """Local object having a .run() method (not LLM pipeline) -> NO REPORT."""
        code = """
class Job:
    def run(self, x): return x*2

j = Job()
y = j.run(3)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 0, "Non-LLM .run() should not be flagged")

    # ========== MIXED CASES ==========

    def test_mixed_calls(self):
        """One structured, one unstructured -> should report only the unstructured one."""
        code = """
import openai

# Unstructured
resp1 = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"A"}]
)

# Structured
resp2 = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"B"}],
    response_format="json"
)
"""
        self.run_rule(code)
        self.assertEqual(len(self.messages), 1, "Should report only the unstructured call")


if __name__ == '__main__':
    unittest.main()
