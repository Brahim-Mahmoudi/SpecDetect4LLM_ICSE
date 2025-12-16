from __future__ import annotations
import os
import ast
import json
import traceback
import argparse
from pathlib import Path
from collections import defaultdict
import sys

RULES_ROOT = Path(__file__).parent / "../Detection/test_rules/"

DETECTION_ROOT = RULES_ROOT.parent

if str(DETECTION_ROOT) not in sys.path:
    sys.path.append(str(DETECTION_ROOT))

def discover_available_rules(rules_root: Path) -> list[str]:
    rule_ids: list[str] = []
    for sub in sorted(rules_root.iterdir()):
        #print(f"Checking {sub}")
        if sub.is_dir() and sub.name.startswith("R"):
            rule_ids.append(sub.name)
    return rule_ids

def import_rule(rule_id: str) -> tuple:
    mod_name = f"test_rules.{rule_id}.generated_rules_{rule_id}"
    module = __import__(mod_name, fromlist=[f"rule_{rule_id}"])
    func = getattr(module, f"rule_{rule_id}")
    return module, func

def analyze_file(filepath: str, selected_rules: list[str]) -> dict[str, list[str]]:
    try:
        source = Path(filepath).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        return {"PARSE_ERROR": [f"Parse error: {e}"]}

    results: dict[str, list[str]] = {}

    for rid in selected_rules:
        module, rule_func = import_rule(rid)
        messages: list[str] = []

        def report(msg):
            messages.append(msg)

        saved_report = getattr(module, "report", None)
        module.report = report

        try:
            add_parent_info = getattr(module, "add_parent_info", None)
            if callable(add_parent_info):
                add_parent_info(tree)

            rule_func(tree)

        except Exception:
            messages.append(f"Execution error:\n{traceback.format_exc()}")

        finally:
            if saved_report is not None:
                module.report = saved_report

        if messages:
            results[rid] = messages

    return results


def analyze_project(root: Path, rules: list[str]) -> tuple[dict[str, dict[str, list[str]]], int]:
    output: dict[str, dict[str, list[str]]] = {}
    total_py_files = 0

    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith(".py"):
                total_py_files += 1
                full = Path(dirpath) / fname
                print(f"Analyzing {full}")
                res = analyze_file(str(full), rules)
                if res:
                    output[str(full)] = res

    return output, total_py_files


def run_analysis(input_dir: Path, selected_rules: list[str]) -> tuple[dict, int, dict]:
    """
    Main function for the web application.
    Returns the results, the total number of files, and the summary.
    """
    
    # Ensure selected rules are valid (optional if Flask validates the UI)
    available = discover_available_rules(RULES_ROOT)
    selected = [r for r in selected_rules if r in available]
    
    if not selected:
        raise ValueError("No valid rules selected.")
        
    # Run the analysis
    results, total_files = analyze_project(input_dir, selected)
    
    # Generate the summary (for display)
    rule_counts = defaultdict(int)
    for file_res in results.values():
        for rule, messages in file_res.items():
            rule_counts[rule] += len(messages)
    
    summary = dict(sorted(rule_counts.items()))
    
    return results, total_files, summary