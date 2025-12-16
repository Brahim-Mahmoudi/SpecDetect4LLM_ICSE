rule R26 "LLM Version Pinning Not Explicitly Set":
    condition:
        exists node in AST: (
            isModelVersionedLLMCall(node) and hasNoModelVersionPinning(node)
        )
    action:
        report "LLM call without model version pinning at line {lineno}"