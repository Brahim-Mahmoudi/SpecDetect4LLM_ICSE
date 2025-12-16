# UMM — Unbounded Max Metrics

## Name & Intent

**Unbounded Max Metrics (UMM)**

Intent: avoid calling LLMs without explicit upper bounds on key resource controls (e.g., `max_output_tokens`, `max_tokens`, input-token budget, timeouts, retries). Explicit bounds reduce error-proneness, keep latency and cost predictable, and prevent memory blowups.

## Context

Hosted LLM APIs expose finite token windows, per-request output caps, and quotas (RPM/TPM/ITPM/OTPM). Systems that ignore these constraints—or omit their own limits on concurrency/response size—are prone to throttling (429s), partial outputs, unstable memory, and runaway latency/costs. This applies across chat/completions/responses, tools/agents, batch jobs, and RAG pipelines.

## Problem

- Unpredictable behavior & costs without hard bounds
- Partial outputs slipping through as "success"
- Inflated memory/instability from unbounded streaming/JSON
- Queue collapse/throughput loss from unbounded timeouts/retries

## Solution

### General Guidelines
1. Bound everything up front:
   - Set `max_output_tokens`
   - Enforce input-token budget
   - Pace requests to quotas
   - Apply timeouts and guards
2. Verify after the call
3. Instrument token usage

### OpenAI Implementation
- Use client-level timeouts and bounded retries
- Pass `max_output_tokens` per request
- Log token usage and finish reasons
- Add alerting on retry spikes

## Effect on Software Quality

### Robustness (RO)
- Fewer truncations/overflow paths
- Safer memory behavior

### Performance (P)
- Predictable latency/throughput
- Controlled cost

### Reliability (R)
- Fewer timeouts/429 storms
- Graceful degradation

### Maintainability (M)
- Explicit, auditable limits
- Easier SRE/playbooks

## Minimal Example (bad → good)

```python
# BAD — Unbounded metrics
client = OpenAI()
resp = client.responses.create(model="gpt-4o-2024-11-20", input=prompt)

# GOOD — Bound tokens, timeouts, retries
from openai import OpenAI, RateLimitError, APITimeoutError, APIError
import time, random

client = OpenAI(timeout=20, max_retries=3)

def generate(prompt: str) -> str:
    MAX_OUT   = 256
    TIMEOUT_S = 20
    MAX_TRIES = 5

    for attempt in range(MAX_TRIES):
        try:
            with client.with_options(timeout=TIMEOUT_S):
                resp = client.responses.create(
                    model="gpt-4.1-mini",
                    input=prompt,
                    max_output_tokens=MAX_OUT
                )
            return resp.output_text

        except RateLimitError as e:
            retry_after = getattr(e, "retry_after", None)
            delay = retry_after or min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(delay)

        except (APITimeoutError, APIError):
            delay = min(8.0, 0.5 * (2 ** attempt))
            time.sleep(delay)

    raise RuntimeError("LLM call failed after bounded retries")
```   
    
### Sources

***Papers***


- Zixi Chen, Yinyu Ye, and Zijie Zhou. 2025. Adaptively Robust LLM Inference Optimization under Prediction Uncertainty. arXiv:2508.14544 [cs.LG] https://arxiv.org/abs/2508.14544

- Tingxu Han, Zhenting Wang, Chunrong Fang, Shunyuan Zhao, Shiqing Ma, and Ziang Chen. 2025. Token-Budget-Aware LLM Reasoning. In ACL 2025. Association for Computational Linguistics, 24842–24855. doi:10.18653/v1/2025.findings-acl.1274

- João Eduardo Montandon, Luciana Lourdes Silva, Cristiano Politowski, Daniel Prates, Arthur de Brito Bonifácio, and Ghizlane El Boussaidi. 2025. Unboxing Default Argument Breaking Changes in Data Science Libraries. (2025). arXiv:2408.05129 [cs.SE] doi:10.48550/arXiv.2408.05129 JSS.

- Zhou, Z., Ning, X., Hong, K., Fu, T., Xu, J., Li, S., ... & Wang, Y. (2024). A survey on efficient inference for large language models. arXiv preprint arXiv:2404.14294.



***Official Documentation***

- [OpenAI — Error codes & rate limits](https://platform.openai.com/docs/guides/error-codes?utm_source=chatgpt.com)

- [Anthropic — Rate limits](https://docs.claude.com/en/api/rate-limits?utm_source=chatgpt.com)

- [Google Gemini — Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits?utm_source=chatgpt.com&hl=fr)

- [Azure OpenAI — Quotas & limits](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/quota?utm_source=chatgpt.com&tabs=rest)

- [Azure API Management — Flexible throttling](https://learn.microsoft.com/en-us/azure/api-management/api-management-sample-flexible-throttling?utm_source=chatgpt.com)

- [Google BigQuery — Quotas & limits](https://cloud.google.com/bigquery/quotas?utm_source=chatgpt.com&hl=fr)


***Engineering Blogs***
- [OpenAI Cookbook — How to handle rate limits?](https://cookbook.openai.com/examples/how_to_handle_rate_limits?utm_source=chatgpt.com)

***Grey Literature***

- [DataDog. LangChain expects at least one chunk from a streaming trace after timeout [Issue #14688]. In dd-trace-py (GitHub repository). GitHub.](https://github.com/DataDog/dd-trace-py/issues/14688)

- [langchain-ai. Streaming inactivity timeout incorrectly aborts after total timeout (@langchain/openai v1.0.0-alpha.1) [Issue #9088]. In langchainjs (GitHub repository). GitHub.](https://github.com/langchain-ai/langchainjs/issues/9088)

- [jcartervi. Add production-ready Pipedrive CRM integration with deduplication and async sync [Pull request #36]. In MyYachtValue (GitHub repository). GitHub.](https://github.com/jcartervi/MyYachtValue/pull/36)

- [jtlicardo. Assistant API WAITING timeout when self-hosting on own instance [Issue #33]. In bpmn-assistant (GitHub repository). GitHub.](https://github.com/jtlicardo/bpmn-assistant/issues/33)
[eslavnov. UMM: Timeout [Issue #19]. In ttmg_server (GitHub repository). GitHub.](https://github.com/eslavnov/ttmg_server/issues/19)

- [RooCodeInc. fix — handle empty stream responses from GLM models [Pull request #8483]. In Roo-Code (GitHub repository). GitHub.](https://github.com/RooCodeInc/Roo-Code/pull/8483)

-[Stack Overflow. RAG — “connection to OpenAI API failed with status: 400 error: -19577 is less than the minimum of 1 — max_tokens” (Accepted answer).](https://stackoverflow.com/questions/77172214/rag-error-connection-to-openai-api-failed-with-status-400-error-19577)

-[Stack Overflow. AI server crashes when I make a request after adding OpenAI code (Accepted answer).](https://stackoverflow.com/questions/77354317/ai-server-crashes-when-i-make-a-request-after-adding-openai-code)