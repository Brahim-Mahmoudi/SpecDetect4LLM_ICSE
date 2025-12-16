# NMVP — No Model Version Pinning

## Name & Intent

**No Model Version Pinning (NMVP)**

Intent: prevent reproducibility and audit gaps that arise when calling an LLM by a moving alias (e.g., `gpt-4o`) rather than an immutable version/snapshot (e.g., `gpt-4o-2024-11-20`). Pinning the exact model version (and recording it with run metadata) stabilizes behavior across time, enables traceability, and supports controlled upgrades through change management. In our empirical study, NMVP appears in 36.00% (72/200) of systems, with a manual-audit precision of 95.65%.

## Context

Most providers and runtimes expose two ways to reference a model:
- A moving alias (e.g., `gpt-4o`, `claude-3-opus`, `llama3.1`), whose underlying weights/prompting/safety layers can change over time
- An immutable version/snapshot (e.g., `gpt-4o-2024-11-20`, a specific Hugging Face revision/commit, an Ollama tag), which pins the exact artifact

Teams often start with aliases during prototyping, but when such references propagate into production (SDK calls, workflow YAML, env vars, router policies), behavior can drift silently as providers roll out updates.

## Problem

Without explicit version pinning, several risks accumulate:

- Silent behavior drift as providers update weights, prompts, or safety filters behind the alias
- Eroded traceability & reproducibility
- Portability issues across environments
- Maintenance overhead for incident analysis and audits
- Difficult rollbacks and change control

Empirically, NMVP generates the most alerts (2,472) yet appears in fewer projects (72/200) than other smells—consistent with alias propagation inside projects once adopted.

## Solution

### General Guidelines
1. Adopt strict version pinning:
   - Use dated model IDs (OpenAI)
   - Use revision commits (Hugging Face)
   - Use image-like tags (Ollama)
2. Govern upgrades via change control
3. Centralize configuration
4. Log & audit
5. Test portability

### OpenAI Implementation
- Prefer dated model IDs (e.g., `gpt-4o-2024-11-20`) over moving aliases
- Pin provider-specific versions in routing layers
- Store model metadata with each run

## Effect on Software Quality

### Maintainability (M)
- Traceability & auditability
- Change control
- Configuration hygiene

### Reliability (R)
- Stable behavior across time/environments
- Comparable evaluations and reproducible outcomes

## Minimal Example (bad → good)

```python
# BAD — Using a moving alias (unPinned)
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o",              # moving alias → behavior can change over time
    messages=messages
)

# GOOD — Pin an immutable, dated model version + log it
from openai import OpenAI
client = OpenAI()

MODEL_ID = "gpt-4o-2024-11-20"   # pinned version/snapshot
resp = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages
    # optional: persist MODEL_ID with other run metadata
)
```
### Sources

***Papers***


- Wilson, G., Aruliah, D. A., Brown, C. T., et al. (2014). Best Practices for Scientific Computing. PLoS Biology. https://pmc.ncbi.nlm.nih.gov/articles/PMC3886731/

- Morishige, M., & Koshihara, R. (2025). Ensuring Reproducibility in Generative AI Systems for General Use Cases: A Framework for Regression Testing and Open Datasets. doi:10.48550/arXiv.2505.02854

- Albertoni, R., Colantonio, S., Skrzypczyński, P., & Stefanowski, J. (2023). Reproducibility of Machine Learning: Terminology, Recommendations and Open Issues. arXiv. https://arxiv.org/abs/2302.12691

- Reyes, F., Gamage, Y., Skoglund, G., Baudry, B., & Monperrus, M. (2024). BUMP: A Benchmark of Reproducible Breaking Dependency Updates. arXiv / SANER 2024. https://arxiv.org/abs/2401.09906

- Venturini, D., Cogo, FR., Polato, I., Gerosa, MA., and Wiese, IS. (2023). I Depended on You and You Broke Me: An Empirical Study of Manifesting Breaking Changes in Client Packages. arXiv:2301.04563 doi:10.48550/arXiv.2301.04563 TOSEM, 2023.

***Official Documentation***

- [Microsoft Learn — Manage foundation model lifecycle (governance & versioning)](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/manage-foundation-models-lifecycle)

- [Hugging Face Transformers — from_pretrained(..., revision=...)](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained.revision)

- [OpenAI — Model pages and release notes (model availability & changes)](https://platform.openai.com/docs/models/gpt-4o)

- [OpenAI — Model pages and release notes (model availability & changes)](https://help.openai.com/en/articles/9624314-model-release-notes)

- [xAI — Models overview](https://docs.x.ai/docs/overview)

- [Google AI — Gemini docs](https://ai.google.dev/)

- [Anthropic — Claude models overview](https://docs.claude.com/en/docs/about-claude/models/overview#model-comparison-table)

- [Ollama — Local model tags](https://ollama.com/)

- [OpenRouter — Multi-provider routing](https://openrouter.ai/)

***Engineering Blogs***
- [Hugging Face — Model Hub (commit snapshots)](https://huggingface.co/)

- [LangChain — Ollama integration (pinning tags in local runtimes)](https://python.langchain.com/docs/integrations/chat/ollama/)

