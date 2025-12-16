![Overview](../static/RQ3.png)

## Add a New Detection Rule

### Integration Workflow
- New rules are added in subfolders of `test_rules/`, following the pattern `test_rules/RX/` where `X` is a rule number **not already used** (1–24 are reserved).
- Each rule must have:
  - A DSL file `test_RX.dsl` defining the rule logic
  - A test file `test_RX.py` for unit validation
  - (Optional for Scenario A) New predicate functions added to `parse.py`

### Scenarios
- **Scenario A**: Define a **new predicate**
  - Add its Python implementation in the `parse.py` header section (as a function taking an AST node and returning a boolean).
  - Use it in your DSL rule condition.

- **Scenario B**: Reuse **existing predicates**
  - Directly write the rule in `test_RX.dsl` using existing semantic predicates.

### Steps to Add a Rule
1. **Create a folder**
   ```bash
   mkdir test_rules/RX/

2. **Write the rule in DSLr**

- Create test_rules/RX/test_RX.dsl. Example:

```
  rule RX "Example rule":
      condition:
          exists node in AST: isExamplePredicate(node)
      action:
          report "Example issue detected"
```
3. ***(Scenario A) Add predicate definition***

- In parse.py, define a new predicate in the list "header". 


- header = [
    ...
  
        "def isExamplePredicate(node):",
          "...",
  ]

4. ***Create unit tests***
- Write test_rules/RX/test_RX.py:

5. ***Run tests***
- pytest test_rules/RX/test_RX.py

All tests must pass to validate the rule.


### Link to Google Form for metrics evaluation 

https://docs.google.com/forms/d/e/1FAIpQLScwzw18u37K6CZvCMFBjJGH2JGBbTSGav0W26slTVBhf2pqRA/viewform?usp=dialog
