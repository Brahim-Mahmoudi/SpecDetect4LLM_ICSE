import os
import io
import re
import ast
import json
import time
import zipfile
import argparse
from typing import Optional, Dict, Any, List

import requests
import pandas as pd


GITHUB_API = "https://api.github.com"
_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def is_full_sha(value: str) -> bool:
    return isinstance(value, str) and _SHA1_RE.match(value.strip()) is not None


def rate_limit_guard(resp: requests.Response) -> None:
    try:
        remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
    except Exception:
        return

    if remaining <= 0 and reset > 0:
        now = int(time.time())
        wait_seconds = max(0, reset - now) + 5
        print(f"Rate limit reached. Sleeping for {wait_seconds}s until the next window.")
        time.sleep(wait_seconds)


def compute_max_depth(node: ast.AST, depth: int = 0) -> int:
    current_max = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            current_max = max(current_max, compute_max_depth(child, depth + 1))
        else:
            current_max = max(current_max, compute_max_depth(child, depth))
    return current_max


def analyze_python_source(source: str, rel_tag: str) -> Optional[Dict[str, Any]]:
    try:
        tree = ast.parse(source, filename=rel_tag)
    except Exception:
        return None

    stats: Dict[str, Any] = {
        "file": rel_tag,
        "num_functions": 0,
        "max_function_depth": 0,
        "num_classes": 0,
        "num_assignments": 0,
        "pipeline_patterns": 0,
        "custom_pipeline_classes": 0,
        "has_dynamic_construct": False,
        "loc": len(source.splitlines()),
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats["num_functions"] += 1

        elif isinstance(node, ast.ClassDef):
            stats["num_classes"] += 1

            name_lower = getattr(node, "name", "").lower()
            is_pipeline = "pipeline" in name_lower

            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if "pipeline" in base_name.lower():
                    is_pipeline = True

            if is_pipeline:
                stats["custom_pipeline_classes"] += 1

        elif isinstance(node, ast.Assign):
            stats["num_assignments"] += 1

        elif isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            func_name_lower = str(func_name).lower()
            if "pipeline" in func_name_lower:
                stats["pipeline_patterns"] += 1

            if func_name in {"getattr", "setattr", "exec", "eval", "__import__"}:
                stats["has_dynamic_construct"] = True

    depths = [
        compute_max_depth(n, depth=1)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    ]
    stats["max_function_depth"] = max(depths) if depths else 0
    return stats


def build_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Python files from GitHub repo zipballs listed in a CSV.")
    parser.add_argument("--csv-file", required=True, help="Path to the CSV file containing at least owner and repo columns.")
    parser.add_argument("--output-dir", default="repo_analyses", help="Directory where JSON outputs will be written.")
    parser.add_argument("--pause-every", type=int, default=100, help="Pause after this many processed repositories.")
    parser.add_argument("--pause-seconds", type=int, default=60, help="Pause duration in seconds.")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable name that contains the GitHub token.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds for zipball downloads.")
    args = parser.parse_args()

    github_token = os.getenv(args.token_env, "").strip() or None

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_file, dtype=str, keep_default_na=False)
    print(f"{len(df)} repositories to process")

    session = requests.Session()
    headers = build_headers(github_token)

    processed = 0

    for idx, row in df.iterrows():
        owner = (row.get("owner") or "").strip()
        repo = (row.get("repo") or "").strip()
        commit_sha = (row.get("commit_sha") or "").strip() or None

        if not owner or not repo:
            print(f"Skipping row {idx + 1}: missing owner or repo")
            continue

        tag = f"{owner}/{repo}" + (f"@{commit_sha[:7]}" if commit_sha else "")
        print(f"\n({idx + 1}/{len(df)}) Analyzing {tag}...")

        json_path = os.path.join(args.output_dir, f"{safe_filename(owner)}__{safe_filename(repo)}.json")
        if os.path.exists(json_path):
            print("JSON already exists, skipping.")
            continue

        zip_url = f"{GITHUB_API}/repos/{owner}/{repo}/zipball"
        if commit_sha:
            if is_full_sha(commit_sha):
                zip_url += f"/{commit_sha}"
            else:
                zip_url += f"/{commit_sha}"

        try:
            resp = session.get(zip_url, headers=headers, timeout=args.timeout)
            if resp.status_code != 200:
                print(f"Failed ({resp.status_code}) for {tag}")
                rate_limit_guard(resp)
                continue
        except Exception as e:
            print(f"Network error for {tag}: {e}")
            continue

        results: List[Dict[str, Any]] = []
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                for file_name in z.namelist():
                    if not file_name.endswith(".py"):
                        continue
                    try:
                        source = z.read(file_name).decode("utf-8", errors="ignore")
                        stats = analyze_python_source(source, f"{owner}/{repo}/{file_name}")
                        if stats:
                            results.append(stats)
                    except Exception:
                        continue
        except zipfile.BadZipFile:
            print(f"Corrupted archive for {tag}")
            continue

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"{tag}: {len(results)} Python files analyzed -> {json_path}")

        processed += 1
        if processed % args.pause_every == 0:
            print(f"Pausing for {args.pause_seconds}s after {processed} repositories...")
            time.sleep(args.pause_seconds)

    print(f"\nDone. {processed} repositories processed. Results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
