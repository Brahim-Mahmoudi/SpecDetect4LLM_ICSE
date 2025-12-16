rule R29 "No Structured Output in Pipeline":
    condition:
        exists node in AST: (
            isUnstructuredLLMCallInPipeline(node) and isTextGeneratingCall(node)
        )
    action: report "LLM pipeline call without structured output at line {lineno}"
