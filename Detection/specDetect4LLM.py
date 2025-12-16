from __future__ import annotations
import os
import ast
import json
import traceback
import argparse
from pathlib import Path
from collections import defaultdict

RULES_ROOT = Path(__file__).parent / "test_rules"


def discover_available_rules(rules_root: Path) -> list[str]:
    rule_ids: list[str] = []
    for sub in sorted(rules_root.iterdir()):
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


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
        description="Run SpecDetect4AI code smell detection on a Python project."
    )
    parser.add_argument(
        "--input-dir", "-i", type=Path, required=False,
        help="Path to the root directory of Python files to analyze"
    )
    parser.add_argument(
        "--output-file", "-o", type=Path, default=Path("specDetectllm_results.json"),
        help="Path to write JSON results (default: specDetect4llm_results.json)"
    )
    parser.add_argument(
        "--rules", "-r", nargs='+', metavar='RULE_ID',
        help="List of rule IDs to run (e.g., R24 R26)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available rules"
    )
    parser.add_argument(
        "--exclude", nargs='+', metavar='RULE_ID',
        help="List of rule IDs to exclude when using --all"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary of results per rule"
    )
    parser.add_argument(
        "--list-rules", action="store_true",
        help="List all available rule IDs and exit"
    )
    args = parser.parse_args()
    print(f"Starting SpecDetect4LLM analysis on project: {args.input_dir} ...")

    available = discover_available_rules(RULES_ROOT)
    if args.list_rules:
        print("Available rules:")
        for r in available:
            print(f"  {r}")
        exit(0)

    if not args.input_dir:
        print("Error: --input-dir is required unless using --list-rules.")
        exit(1)

    if args.all:
        selected = [r for r in available if not args.exclude or r not in args.exclude]
    elif args.rules:
        selected = args.rules
    else:
        print("Error: You must specify either --rules or --all.")
        exit(1)

    invalid = [r for r in selected if r not in available]
    if invalid:
        print(f"Error: Unknown rule IDs: {invalid}\nUse --list-rules to see available rules.")
        exit(1)

    print(f"Analyzing {args.input_dir} with rules: {selected}")
    print("_________")

    results, total_files = analyze_project(args.input_dir, selected)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("           ")
    print(f"Results written to {args.output_file} ({len(results)} files with alerts out of {total_files} total .py files)")

    if args.summary:
        print("\n___Summary by rule___")
        rule_counts = defaultdict(int)
        for file_res in results.values():
            for rule, messages in file_res.items():
                rule_counts[rule] += len(messages)
        for rule, count in sorted(rule_counts.items()):
            print(f"  {rule}: {count} alerts")
