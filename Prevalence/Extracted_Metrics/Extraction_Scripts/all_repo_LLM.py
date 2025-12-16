#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for specDetect4ai on multiple projects with rules R25..R29.

Usage example:
  python batch_specdetect.py \
    --specdetect /path/to/specDetect4ai.py \
    --input-root /path/to/projects_root \
    --output-dir /path/to/results \
    --jobs 4 --timeout 900 --summary

- Discovers projects as immediate subdirectories of --input-root (can be changed with --depth).
- A project is considered valid if it contains at least one .py file recursively.
- Writes one JSON per project in --output-dir: <project_name>.json
"""

from __future__ import annotations  # <-- doit être ici

import argparse
import concurrent.futures as cf
import json
import os
import sys
import subprocess
import time
from pathlib import Path

DEFAULT_RULES = ["R25", "R26", "R27", "R28", "R29"]

def has_py_files(path: Path) -> bool:
    try:
        for _ in path.rglob("*.py"):
            return True
    except Exception:
        return False
    return False

def discover_projects(root: Path, depth: int) -> list[Path]:
    """
    Returns a list of project directories under root up to `depth` levels.
    Depth=1 -> only direct subfolders.
    """
    if depth < 1:
        depth = 1
    projects = []
    # BFS traversal up to 'depth' directories down
    queue = [(root, 0)]
    seen = set()
    while queue:
        base, d = queue.pop(0)
        if not base.exists() or not base.is_dir():
            continue
        try:
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                # Skip VCS / virtualenv / cache folders
                name = child.name
                if name.startswith(".") or name in {"__pycache__", ".git", ".hg", ".svn", ".venv", "venv"}:
                    continue
                if d + 1 < depth:
                    queue.append((child, d + 1))
                # Consider this dir as a project if it has at least one .py
                if has_py_files(child):
                    projects.append(child)
        except PermissionError:
            continue
    # Remove duplicates while preserving order
    unique = []
    seen = set()
    for p in projects:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def run_one(specdetect: Path, project: Path, out_dir: Path, rules: list[str], summary: bool, timeout: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{project.name}.json"
    cmd = [
        sys.executable,
        str(specdetect),
        "--input-dir", str(project),
        "--rules", *rules,
        "--output-file", str(out_json),
    ]
    if summary:
        cmd.append("--summary")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
        )
        dt = time.time() - t0
        ok = proc.returncode == 0
        return {
            "project": str(project),
            "output": str(out_json),
            "ok": ok,
            "returncode": proc.returncode,
            "elapsed_s": round(dt, 2),
            "stdout": proc.stdout[-2000:],  # tail for brevity
            "stderr": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "project": str(project),
            "output": str(out_json),
            "ok": False,
            "returncode": None,
            "elapsed_s": timeout,
            "stdout": (e.stdout or "")[-2000:],
            "stderr": f"Timeout after {timeout}s",
        }
    except Exception as e:
        return {
            "project": str(project),
            "output": str(out_json),
            "ok": False,
            "returncode": None,
            "elapsed_s": round(time.time() - t0, 2),
            "stdout": "",
            "stderr": f"Exception: {e}",
        }

def parse_args():
    ap = argparse.ArgumentParser(description="Batch runner for specDetect4ai on multiple projects with R25..R29.")
    ap.add_argument("--specdetect", required=True, type=Path,
                    help="Chemin vers specDetect4ai.py")
    ap.add_argument("--input-root", required=True, type=Path,
                    help="Dossier racine contenant les projets (chaque sous-dossier = un projet).")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Dossier de sortie pour les JSON générés.")
    ap.add_argument("--rules", nargs="+", default=DEFAULT_RULES,
                    help="Liste de règles à appliquer (défaut: R25 R26 R27 R28 R29).")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="Nombre de tâches en parallèle (défaut: nb de CPU).")
    ap.add_argument("--timeout", type=int, default=0,
                    help="Timeout par projet en secondes (0 = illimité).")
    ap.add_argument("--summary", action="store_true",
                    help="Afficher un résumé CLI pour chaque exécution.")
    ap.add_argument("--depth", type=int, default=1,
                    help="Profondeur de découverte (1 = sous-dossiers immédiats).")
    ap.add_argument("--force", action="store_true",
                    help="Recalculer même si le JSON existe déjà.")
    return ap.parse_args()

def main():
    args = parse_args()

    specdetect = args.specdetect.resolve()
    if not specdetect.exists():
        print(f"[ERREUR] specDetect4ai.py introuvable: {specdetect}", file=sys.stderr)
        sys.exit(2)

    input_root = args.input_root.resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"[ERREUR] input-root invalide: {input_root}", file=sys.stderr)
        sys.exit(2)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = discover_projects(input_root, depth=args.depth)
    if not projects:
        print("[INFO] Aucun projet Python trouvé sous:", input_root, file=sys.stderr)
        sys.exit(0)

    # Filter out those already processed if not --force
    to_run = []
    for p in projects:
        out_json = output_dir / f"{p.name}.json"
        if out_json.exists() and not args.force:
            print(f"[SKIP] {p.name} (résultat déjà présent)")
            continue
        to_run.append(p)

    if not to_run:
        print("[INFO] Rien à faire (tous les JSON existent déjà).")
        sys.exit(0)

    print(f"[INFO] Projets à traiter: {len(to_run)} / {len(projects)}")
    print(f"[INFO] Règles: {' '.join(args.rules)}")
    print(f"[INFO] Sortie: {output_dir}")

    results = []
    # Parallel execution
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        fut2proj = {
            ex.submit(run_one, specdetect, proj, output_dir, args.rules, args.summary, args.timeout): proj
            for proj in to_run
        }
        for fut in cf.as_completed(fut2proj):
            res = fut.result()
            results.append(res)
            status = "OK" if res["ok"] else "FAIL"
            print(f"[{status}] {Path(res['project']).name} -> {res['output']} (t={res['elapsed_s']}s)")
            if not res["ok"]:
                print("  stderr tail:", res["stderr"][:500].replace("\n", " "))
    
    # Write a batch summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Rapport global: {summary_path}")

    # Exit code: number of failures
    failures = sum(1 for r in results if not r["ok"])
    sys.exit(failures)

if __name__ == "__main__":
    main()

