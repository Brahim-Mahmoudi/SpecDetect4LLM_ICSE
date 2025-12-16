rule R27 "LLM With No System Message":
    condition:
        exists node in AST: (
            isRoleBasedLLMChat(node) and hasNoSystemMessage(node)
        )
    action:
        report "LLM chat call without a system message at line {lineno}"