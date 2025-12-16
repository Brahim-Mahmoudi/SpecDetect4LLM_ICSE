
import os, json, math, argparse, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def load_repo_jsons(input_dir: Path) -> List[Tuple[str, str, list]]:
    """Return list of (owner, repo, records) where records is a list of per-file dicts."""
    items = []
    for p in input_dir.glob("*.json"):
        name = p.stem  # expect owner__repo
        if "__" in name:
            owner, repo = name.split("__", 1)
        else:
            owner, repo = "unknown", name
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items.append((owner, repo, data))
        except Exception:
            continue
    return items

def aggregate_repo(owner: str, repo: str, records: list) -> Dict[str, Any]:
    """Aggregate per-file metrics into project-level metrics."""
    total_loc = 0
    files = 0
    total_functions = 0
    total_classes = 0
    total_assignments = 0
    total_pipelines = 0
    total_custom_pipeline_classes = 0
    max_function_depth_global = 0
    has_dynamic_any = False

    for r in records:
        loc = int(r.get("loc", 0) or 0)
        total_loc += loc
        files += 1
        total_functions += int(r.get("num_functions", 0) or 0)
        total_classes += int(r.get("num_classes", 0) or 0)
        total_assignments += int(r.get("num_assignments", 0) or 0)
        total_pipelines += int(r.get("pipeline_patterns", 0) or 0)
        total_custom_pipeline_classes += int(r.get("custom_pipeline_classes", 0) or 0)
        max_function_depth_global = max(max_function_depth_global, int(r.get("max_function_depth", 0) or 0))
        has_dynamic_any = has_dynamic_any or bool(r.get("has_dynamic_construct", False))

    functions_per_kloc = (total_functions / total_loc * 1000) if total_loc > 0 else 0.0
    assignments_per_kloc = (total_assignments / total_loc * 1000) if total_loc > 0 else 0.0
    pipelines_per_kloc = (total_pipelines / total_loc * 1000) if total_loc > 0 else 0.0

    return {
        "owner": owner,
        "repo": repo,
        "files_py": files,
        "total_loc": total_loc,
        "total_functions": total_functions,
        "total_classes": total_classes,
        "total_assignments": total_assignments,
        "total_pipelines": total_pipelines,
        "total_custom_pipeline_classes": total_custom_pipeline_classes,
        "max_function_depth": max_function_depth_global,
        "has_dynamic_construct_any": has_dynamic_any,
        "functions_per_kloc": round(functions_per_kloc, 3),
        "assignments_per_kloc": round(assignments_per_kloc, 3),
        "pipelines_per_kloc": round(pipelines_per_kloc, 3),
        "functions_per_file": round((total_functions / files), 3) if files > 0 else 0.0,
    }

def compute_complexity_score(df: pd.DataFrame) -> pd.Series:
    """Composite complexity: depth + density + pipelines + dynamic flag."""
    # Normalize with min-max (robust to zeros). Add small eps to avoid division by zero.
    def minmax(x):
        if x.max() == x.min():
            return pd.Series([0.5] * len(x), index=x.index)
        return (x - x.min()) / (x.max() - x.min())

    depth = minmax(df["max_function_depth"].fillna(0))
    fpf = minmax(df["functions_per_file"].fillna(0))
    pip = minmax(df["pipelines_per_kloc"].fillna(0))
    dyn = df["has_dynamic_construct_any"].astype(int)

    # Weights can be tuned
    score = 0.4*depth + 0.25*fpf + 0.25*pip + 0.1*dyn
    return score

def size_buckets_by_quantiles(df: pd.DataFrame) -> pd.Series:
    """Bucket by total_loc terciles: small (Q0-Q33), medium (Q33-Q66), large (Q66-Q100)."""
    if df["total_loc"].nunique() == 1:
        return pd.Series(["medium"]*len(df), index=df.index)
    q33 = df["total_loc"].quantile(0.33)
    q66 = df["total_loc"].quantile(0.66)
    def bucket(v):
        if v <= q33: return "small"
        if v <= q66: return "medium"
        return "large"
    return df["total_loc"].apply(bucket)

def pick_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Pick 2 per bucket: one low-complexity, one high-complexity."""
    picks = []
    for b in ["small", "medium", "large"]:
        sub = df[df["size_bucket"] == b].copy()
        if sub.empty:
            continue
        sub_sorted = sub.sort_values("complexity_score")
        low = sub_sorted.head(1)
        high = sub_sorted.tail(1)
        # If only one project in bucket, low==high; drop duplicate by repo
        picks.append(low)
        if not high.iloc[0]["repo"] == low.iloc[0]["repo"]:
            picks.append(high)
    if picks:
        return pd.concat(picks, ignore_index=True)
    return pd.DataFrame(columns=df.columns)

def main():
    ap = argparse.ArgumentParser(description="Select repos for manual evaluation from per-repo JSONs.")
    ap.add_argument("--input", required=True, help="Folder containing per-repo JSON files (one JSON per repo).")
    ap.add_argument("--out", required=True, help="Output folder for CSV summaries and selection.")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_repo_jsons(input_dir)
    if not items:
        print(f"No JSON files found in {input_dir}. Expect files like owner__repo.json")
        return

    rows = []
    for owner, repo, records in items:
        agg = aggregate_repo(owner, repo, records)
        rows.append(agg)

    df = pd.DataFrame(rows)
    # Exclude repos with less than 10 Python files or zero LOC (no analyzable content)
    excluded = df[(df['files_py'] <= 0) | (df['total_loc'] <= 0)].copy()
    excluded['reason'] = excluded.apply(lambda r: 'no_python_files' if r['files_py'] <= 0 else 'zero_loc', axis=1)
    df = df[(df['files_py'] > 0) & (df['total_loc'] > 0)].copy()
    # Save excluded list for transparency
    excl_csv = out_dir / 'excluded_repos.csv'
    if not excluded.empty:
        excluded.to_csv(excl_csv, index=False)
    # Guard: if nothing remains, stop early
    if df.empty:
        print('All repos were excluded (no Python files or zero LOC). See excluded_repos.csv.')
        return
    # Exclude repos with zero Python files or zero LOC (no analyzable content)
    excluded = df[(df["files_py"] <= 10) | (df["total_loc"] <= 10)].copy()
    excluded["reason"] = excluded.apply(lambda r: "no_python_files" if r["files_py"] <= 0 else "zero_loc", axis=1)
    df = df[(df["files_py"] > 10) & (df["total_loc"] > 10)].copy()
    # Save excluded list for transparency
    excl_csv = out_dir / "excluded_repos.csv"
    if not excluded.empty:
        excluded.to_csv(excl_csv, index=False)
    # Guard: if nothing remains, stop early
    if df.empty:
        print("All repos were excluded (no Python files or zero LOC). See excluded_repos.csv.")
        return
    # Compute derived metrics
    df["complexity_score"] = compute_complexity_score(df).round(3)
    df["size_bucket"] = size_buckets_by_quantiles(df)

    # Save full summary
    summary_csv = out_dir / "repos_summary.csv"
    df.sort_values(["size_bucket","complexity_score"], inplace=True)
    df.to_csv(summary_csv, index=False)

    # Pick candidates
    picks = pick_candidates(df)
    picks_csv = out_dir / "selection_candidates.csv"
    picks.to_csv(picks_csv, index=False)

    # Also produce a lightweight markdown summary for copy-paste
    md_lines = ["# Suggested Repos for Manual Evaluation\n"]
    for b in ["small","medium","large"]:
        sub = picks[picks["size_bucket"]==b]
        if sub.empty: 
            continue
        md_lines.append(f"## {b.title()}")
        for _, r in sub.iterrows():
            md_lines.append(f"- **{r['owner']}/{r['repo']}** — LOC: {int(r['total_loc'])}, Files: {int(r['files_py'])}, Depth: {int(r['max_function_depth'])}, Pipelines/kloc: {r['pipelines_per_kloc']}, Dyn: {bool(r['has_dynamic_construct_any'])}, Complexity: {r['complexity_score']}")
        md_lines.append("")
    (out_dir / "selection_candidates.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f" Wrote full summary: {summary_csv}")
    print(f" Wrote picks:        {picks_csv}")
    print(f" Markdown summary:   {out_dir/'selection_candidates.md'}")
    print("\nExample next steps:")
    print(" - Manually inspect the selected repos to build ground truth for precision/recall.")
    print(" - You can adjust bucket thresholds or complexity weights if needed.")

if __name__ == "__main__":
    main()
