rule R25 "LLM Temperature Not Explicitly Set":
    condition:
        exists node in AST: (
            isLLMCallRequiringTemperature(node) and hasNoTemperatureParameter(node) and not (hasParameterSet(node, "top_p") or hasParameterSet(node, "top_k"))
        )
    action:
        report "LLM call without explicit temperature parameter at line {lineno}"