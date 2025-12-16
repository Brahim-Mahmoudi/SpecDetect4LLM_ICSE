# Methodology: Literature Review & Code-Smell Extraction (LLM Integration)

This document details how we (i) searched and selected the literature on LLM integration code smells  and (ii) constructed the resulting code-smell catalog.

### Here are the code smells descriptions :

- [No_Structured_Output](Code_Smells_Description/No_Structured_Output.md)
- [No_System_Message](Code_Smells_Description/No_System_Message.md)
- [No_Version_Model_Pinning](Code_Smells_Description/No_Version_Model_Pinning.md)
- [Temperature_Not_Explicitly_Set](Code_Smells_Description/Temperature_Not_Explicitly_Set.md)
- [Unbounded_Max_Metrics](Code_Smells_Description/Unbounded_Max_Metrics.md)


## 1) Scope & Research Questions

We target code-level issues that arise when integrating Large Language Models (LLMs) into software systems (e.g., API misuse, quota handling, structured output, prompt plumbing, reliability, cost/performance risks). Our review asks:

- **RQ1 — Evidence**: What defects, smells, or anti-patterns are reported around LLM/AI integration?
- **RQ2 — Practices**: What mitigation strategies, refactorings, or best practices are suggested?
- **RQ3 — Catalog**: How can these findings be consolidated into a practitioner-oriented code smell catalog?



## 2) Data Sources

- **Academic**: Google Scholar, ACM Digital Library, IEEE Xplore, arXiv, Scopus.
- **Grey literature**: provider docs, engineering blogs, cookbooks, issue trackers, Q&A, and technical posts.




## 3) Literature Analysis Methodology

### Manual Literature Search in Academic Sources

The first step was a manual bibliographic search across established academic portals. Simple queries were issued on platforms such as Google Scholar, IEEE Xplore, and the ACM Digital Library, as well as open repositories like arXiv. These searches aimed to gather publications that address—directly or indirectly—code-level LLM integration. Because the notion of "LLM code smell" was not yet formalized, we employed a broad set of keywords around defects, poor practices, and recommendations for using LLMs in software. Examples include "LLM integration issues", "LLM best practices in code", and "pitfalls of using LLM APIs".

This manual search produced an initial corpus of potentially relevant academic articles. At this stage, the emphasis was on coverage rather than precision: we preferred to cast a wide net to avoid missing important sources. The process was entirely manual and qualitative—no automated crawling scripts were used. The goal was to prioritize deep understanding of content over raw volume.

### Selection and Filtering of Relevant Sources

After assembling the initial list from the queries, we manually screened sources for relevance. Each result was examined by reading its title and abstract to quickly assess whether it addressed concrete coding or software engineering aspects of LLM integration. Selection criteria included:

- The source explicitly discussed LLM integration in code (e.g., LLM API calls, parameter configurations, architectural concerns around LLMs).
- Ideally, it identified concrete problems, limitations, or pitfalls of LLM integration, or offered actionable coding recommendations.
- Work that was purely conceptual or too high-level was excluded.

Applying these criteria, we refined the initial corpus and retained the most relevant articles for in-depth reading. This manual triage narrowed the set to studies and reports likely to reveal recurrent poor coding practices when integrating LLMs in real projects.

### In-Depth Reading and Analysis of Selected Sources

For each retained source, we performed a full, in-depth reading. This qualitative analysis had two aims:
1. Extract any passages describing a difficulty, frequent mistake, or coding advice related to LLM implementation
2. Identify recurring themes across sources

We kept structured notes per reference, recording the problems mentioned (e.g., "not specifying temperature can cause issues"), their context, and any suggested best practices or fixes.

As this analytical reading progressed, candidate code smells emerged. Whenever a poor practice was mentioned across multiple independent sources, we treated it as a credible code smell rather than an isolated anecdote.

### Grey Literature and Triangulation

To enrich and validate observations, we also consulted grey literature:
- Technical blog posts
- Developer write-ups
- Official documentation from LLM API providers
- Community experience reports

The goal was to triangulate perspectives: academic publications provide structured, systematic views of problems, while grey literature reflects practical developer concerns and field-tested guidance.

This triangulation was essential to validate each candidate code smell. For every poor practice identified in academic sources, we looked for supporting evidence in blogs or documentation (and vice versa). Items mentioned by only a single source were treated cautiously or discarded if they lacked corroboration.

### Final Extraction and Formalization of Code Smells

Following analysis and triangulation, we identified five recurrent poor practices in LLM integration. Each smell was then formalized and documented with:

- **Name & Intent**
- **Context**
- **Problem**
- **Solution**
- **Effect on Software Quality**
- **Minimal Example (bad → good)**
- **Sources/References**

This documentation structure follows established code smell catalogs in software engineering, facilitating developer uptake. The five smells obtained from this manual, triangulated analysis provide a first formal basis to recognize and avoid poor coding practices in LLM integration.





## 4) Threats to Validity

- Search bias: mitigated via multiple portals, synonym expansion, and snowballing.
- Grey-literature credibility: mitigated by favoring provider docs and well-established engineering sources.
- Evolving APIs: vendor limits and SDKs change; we timestamp sources and note model/API versions.
- Construct validity: our smells target code-integration phenomena (not prompt quality alone); definitions underwent expert review.
