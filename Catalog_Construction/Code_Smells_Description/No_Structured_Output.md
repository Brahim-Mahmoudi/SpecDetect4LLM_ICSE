# NSO — No Structured Output

## Name & Intent

**No Structured Output (NSO)**. 

***Intent***: prevent consuming free-form LLM text where typed fields (e.g., JSON) are expected by enforcing and validating an output schema strictly at the boundary before any parsing, indexing, execution, or storage. In our empirical study, NSO affects 40.50% (81/200) of systems with 80.00% precision in the audited sample.

***Context***

NSO arises when LLM-integrating systems expect structured/typed outputs but instead rely on free-form inference text in their prompt and subsequently parse/index/execute it as structured content. This is common across assistants/agents, data pipelines, automation inferences and any integration that later treats the model response as structured data.

***Problem***

Without an enforced output schema, the system can receive free-form text where structured fields are required, leading to:
- Schema drift
- Missing/renamed fields
- Type mismatches
- Silent truncation mistaken for success
- Breaking parsers and downstream steps
- Reliability degradation as runs become inconsistent
- Stores accumulating corrupted/hallucinated values
- Execution/storage/display paths facing higher injection risks

***Solution***

Enforce structured output at the API boundary and validate the response's format before use.

***OpenAI Implementation***
- Declare JSON Schema via `response_format` (chat completions) or `text.format` (responses)
- Alternatively Use Python SDK to bind formats to classes
- Always validate results to handle refusals/errors



## Effect on Software Quality

### Robustness (RO)
- Error-prone behavior from schema drift
- Type mismatches
- Truncation that "passes" as success

### Reliability (R)
- Inconsistent runs
- Corrupted/hallucinated values in stores
- Injection risks along execution/storage/display paths

## Minimal Example (bad → good)

```python
# BAD — Free-form output; no schema; brittle parsing
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=[{"role": "user", "content": "Return user profile"}]
)
text = resp.choices[0].message.content
# Downstream code *assumes* JSON and may crash or silently misparse.
# data = json.loads(text)

# GOOD — Enforce JSON Schema + strict parse/validate at the boundary
from openai import OpenAI
import json, jsonschema

client = OpenAI()

user_schema = {
    "type": "object",
    "required": ["id", "name", "email"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 1},
        "email": {"type": "string"}
    },
    "additionalProperties": False
}

resp = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=[{"role": "user", "content": "Return user profile"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "UserRecord", "schema": user_schema}
    }
)
payload = resp.choices[0].message  # SDK returns JSON matching the schema
data = json.loads(payload.content)
jsonschema.validate(instance=data, schema=user_schema)  # defense-in-depth
```



### Sources

***Papers***

- Michael Xieyang Liu, Frederick Liu, Alexander J. Fiannaca, Terry Koo, Lucas Dixon, Michael Terry, and Carrie J. Cai. 2024. "We Need Structured Output":Towards User-centered Constraints on Large Language Model Output. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI EA ’24). Association for Computing Machinery, New York, NY, USA, Article 10, 9 pages. doi:10.1145/3613905.3650756

***Official Documentation***

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Azure OpenAI Structured Outputs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs)
- [Dev.to (Pydantic)](https://dev.to/devasservice/a-practical-guide-on-structuring-llm-outputs-with-pydantic-50b4)
- [Kharitonov (TDS/Medium)](https://medium.com/data-science/enforcing-json-outputs-in-commercial-llms-3db590b9b3c8)

***Engineering Blogs***
- [Kharitonov, Enforcing JSON Outputs in Commercial LLMs](https://medium.com/data-science/enforcing-json-outputs-in-commercial-llms-3db590b9b3c8)
- [Dev.to, Structuring LLM Outputs with Pydantic](https://dev.to/devasservice/a-practical-guide-on-structuring-llm-outputs-with-pydantic-50b4)
- [Okareo, Validate the Output of LLM-Based Products](https://okareo.com/blog/posts/validate-llm-output)
- [Modelmetry, Ensure LLM Output Adheres to a JSON Schema](https://modelmetry.com/blog/how-to-ensure-llm-output-adheres-to-a-json-schema)

***Grey Literature***

- [dgy516. test(functional) — streaming JSON missing required; completions logprobs structure [Pull request #60]. In vllm_cibench (GitHub repository). GitHub.]( https://github.com/dgy516/vllm_cibench/pull/60)

- [vllm-project. Structured output is not correctly enforced when using GPT-OSS [Issue #23120]. In vllm (GitHub repository). GitHub. ](https://github.com/vllm-project/vllm/issues/23120)

- [Microsoft. Python — OpenAI Responses client structured output does not work with streaming responses [Issue #238]. In agent-framework (GitHub repository). GitHub.]( https://github.com/microsoft/agent-framework/issues/238)

- [BerriAI. Responses — add structured output for SDK [Pull request #14206]. In litellm (GitHub repository). GitHub.](https://github.com/BerriAI/litellm/pull/14206)

