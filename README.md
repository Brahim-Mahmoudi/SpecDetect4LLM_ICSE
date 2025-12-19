![Overview](static/MethoLLMP.png)

# LLM integration Code Smells — Replication Package

> Companion materials for **specifying**, **detecting**, and **measuring the prevalence** of *LLM integration code smells*.



---


## How to use **SpecDetect4LLM**: 

- [Web-app](Detection/docs/docker.md) (Recommended (Docker we app))
- [Command Line](/Detection/docs/usage.md) 



## 1) `Catalog_Construction`

- [Catalog Construction](/Catalog_Construction/README.md) 

This folder contains the **formal specification** of each LLM code smell, including:

-**Smell_Extraction**, all the procedures and values extracted to create the code smell catalog.

and, for each code smell:
- **Name & Intent**
- **Context**
- **Problem**
- **Solution**
- **Effect on Software Quality**
- **Minimal Example (bad → good)**
- **Sources/References**



> Use these files to understand the semantics, rationale, and expected fixes for each code smell.

---

## 2) `Detection`

This folder provides:
- **SpecDetect4LLM**, with the **new detection rules** for LLM integration in `/test_rules`. 

---

## 3) `Prevalence`

This folder provides:
- The **dataset** used in our study in Dataset
- **Precision Calculation** (manual detection and precision calculation)

    Run
        `python Prevalence/Precision_Calculation/compute_precision.py`

    Outputs
    The script writes:
        `precision_by_smell.csv` in the current working directory
        and prints a summary to stdout including micro average and macro average precision
- **Extracted metrics** (complexity and prevalence metrics)
- Generated **charts/figures** (PNG)

