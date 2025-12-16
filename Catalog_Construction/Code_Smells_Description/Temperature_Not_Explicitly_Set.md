# TNES — LLM Temperature Not Explicitly Set

## Name & Intent

**LLM Temperature Not Explicitly Set (TNES)**

Intent: avoid relying on a provider/model/framework default temperature when invoking an LLM. Temperature controls the randomness of token sampling; leaving it implicit harms reproducibility, portability, and consistency because defaults vary across providers and can change over time. Projects should make temperature explicit, document it, and tune it by task (lower for precision/repeatability, higher for creativity).

## Context

Modern LLM stacks (APIs, SDKs, local runtimes) expose temperature as a first-class decoding parameter (e.g., OpenAI, Hugging Face/Transformers, Ollama). In practice, teams often prototype without setting temperature, then that omission persists into agents, batch jobs, or evaluation pipelines, where stable behavior over time matters. Because defaults differ across providers/models—and may be revised—leaving temperature implicit pushes hidden choices into the platform, weakening traceability and auditability of runs.

## Problem

Not explicitly setting the temperature creates several quality risks:

- Maintainability & reliability regressions due to different defaults
- Reduced reproducibility & portability
- Hidden changes over time from platform updates
- Undermined stability and audits

## Solution

### General Guidelines
Always set temperature explicitly and document the choice. Tune by task:
- Low (≈0–0.3) for precise, repeatable automation
- Higher (≈0.7–1.0) for creative/divergent generation
- Avoid extremes that degrade coherence

### OpenAI Implementation
- Provide explicit temperature in every call
- Treat temperature as part of run configuration
- Keep temperature conservative with structured outputs

## Effect on Software Quality

### Maintainability (M)
- Configuration is explicit and versionable
- Reduces drift across environments/providers

### Reliability (R)
- Consistent behavior over time/runs
- Fewer surprises from default changes

## Minimal Example (bad → good)

```python
# BAD — Omits temperature (smell)
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=messages
)

# GOOD — Make temperature explicit + treat it as config
from openai import OpenAI
client = OpenAI()

TEMP = 1.0  # choose per task; keep low (≈0–0.3) for repeatability, higher for creativity
resp = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=messages,
    temperature=TEMP
)
# Log TEMP with other run metadata for traceability
```

### Sources

***Papers***


- Montandon, R., et al. (2024). Default parameters can change over time (on instability from shifting defaults). arXiv:2408.05129. https://arxiv.org/pdf/2408.05129

- OpenReview (2020). On high temperatures and incoherence in language generation. https://openreview.net/forum?id=FBkpCyujtS (PDF: https://openreview.net/pdf?id=FBkpCyujtS)



***Official Documentation***

- [OpenAI — Chat Completions API (temperature parameter)](https://platform.openai.com/docs/api-reference/chat/create)

- [Anthropic/Claude — Messages API (temperature)](https://docs.claude.com/en/api/messages)

- [Google — Gemini API (temperature)](https://ai.google.dev/api/generate-content?hl=en)

- [Ollama — Modelfile defaults (temperature)](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)

- [xAI — API reference (chat completions, temperature)](https://docs.x.ai/docs/api-reference#chat-completions)

- [Hugging Face Transformers — Generation config (temperature)](https://huggingface.co/docs/transformers/main_classes/text_generation)


***Engineering Blogs***
- [Vellum — LLM Temperature: How It Works and When You Should Use It](https://www.vellum.ai/llm-parameters/temperature)

- [IBM Think — LLM temperature definition and task guidance](https://www.ibm.com/think/topics/llm-temperature)


***Grey Literature***

- [vllm-project. MCP-USE with VLLM gpt-oss:20b via ChatOpenAI [Issue #26806]. In vllm (GitHub repository). GitHub](https://github.com/vllm-project/vllm/issues/26806)

- [langfuse. Bug — When streaming responses with the OpenAI Responses API, temperature is not captured correctly [Issue #9566] In langfuse (GitHub repository). GitHub.](https://github.com/langfuse/langfuse/issues/9566)