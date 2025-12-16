##  Usage

```bash
python specDetect4LLM.py --input-dir <path> [--output-file <file>] [--rules R2 R6] [--all] [--exclude R6] [--summary] [--list-rules]
```

##  Examples

### Analyze a folder using all rules
```bash
python specDetect4LLM.py --input-dir ./my_project --all
```

### Analyze with specific rules only
```bash
python specDetect4LLM.py --input-dir ./my_project --rules R2 R6 R11
```

### Analyze all rules except some
```bash
python specDetect4LLM.py --input-dir ./my_project --all --exclude R6 R11
```

### Save results to a custom file
```bash
python specDetect4LLM.py --input-dir ./my_project --all --output-file results.json
```

### List all available rules
```bash
python specDetect4LLM.py --list-rules
```

### Show a summary of alerts by rule
```bash
python specDetect4LLM.py --input-dir ./my_project --all --summary
```

### Run all tests
```bash
./run_all_tests.sh
```

##  Output
- A JSON file containing detected issues, grouped by file and rule
- CLI printout showing:
  - Number of `.py` files scanned
  - Number of files with alerts
  - (Optional) Summary of alerts per rule

##  Project Structure Assumptions
- Rules are located in subfolders of `test_rules/` (e.g. `test_rules/R2/generated_rules_R2.py`)
- Each rule module must define a function named `rule_<ID>` and a global `report()` function used for collecting messages.

![Overview](../static/DSL.png)

##  Requirements
- Python 3.9+
- Lark 1.2.2
- python -m pip install -r requirements.txt
- The `test_rules/` directory must exist and contain valid rule implementations
### Example : 

- python -m venv .venv
- source .venv/bin/activate  # or .venv\\Scripts\\activate sur Windows
- python -m pip install -r requirements.txt


##  Notes

- The number of analyzed files includes all `.py` files recursively found under `--input-dir`
