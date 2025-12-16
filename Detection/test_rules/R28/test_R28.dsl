rule R28 "LLM Calls Without Bounded Metrics":
    condition:
        exists node in AST: (
            isLLMCall(node) and hasNoBoundedMetrics(node) and isNotSDKClient(node)
        )
    action:
        report "LLM call without bounded metrics (tokens/timeout/response size) at line {lineno}"