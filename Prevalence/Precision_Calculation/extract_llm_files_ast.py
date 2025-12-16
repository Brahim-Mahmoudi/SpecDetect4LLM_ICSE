#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Python files that contain LLM usages (OpenAI, Anthropic, HF Transformers, Ollama, vLLM, etc.)
from a list of GitHub repositories. Detection uses AST (not just regex) and outputs:
 - A folder per repo with only the matching .py files
 - A global CSV summary (per-file)
 - A per-repo JSON metadata file listing matches (file, provider, lines)

Usage:
  python extract_llm_files_ast.py     --csv /path/to/merged_repos.csv     --out /path/to/LLM_Files_Extracted     --summary /path/to/llm_extraction_summary.csv     --token YOUR_GITHUB_TOKEN     [--pause-every 80] [--pause-seconds 60]

The input CSV must have columns: owner, repo, (optional) commit_sha
"""

import argparse, io, json, os, re, sys, time, zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
import ast

# ------------------------- Configurable provider signatures -------------------------

PROVIDER_MODULES = {
    "openai": {"openai"},
    "anthropic": {"anthropic"},
    "transformers": {"transformers"},
    "ollama": {"ollama"},
    "vllm": {"vllm"},
    "llama_cpp": {"llama_cpp", "llama_cpp_python"},
    "google_genai": {"google.generativeai", "google.ai.generativelanguage", "vertexai"},
    "openrouter": {"openrouter"},
    "langchain": {"langchain", "langchain_core", "langchain_openai", "langchain_community"},
    "llama_index": {"llama_index", "gpt_index"},
    "groq": {"groq"},
    "mistralai": {"mistralai"},
    "cohere": {"cohere"},
    "google_genai": {"google.generativeai", "vertexai", "google.ai.generativelanguage"},
    "bedrock": {"boto3"},  # Amazon Bedrock client
    }

# Call attribute tails frequently used for LLM inference (we detect these on attribute chains)
CALL_TAILS = {
    "openai": {
        ("responses", "create"),
        ("chat", "completions", "create"),
        ("completions", "create"),
        ("batches", "create"),
        ("responses", "parse"),
    },
    "anthropic": {
        ("messages", "create"),
        ("completions", "create"),
    },
    "transformers": {
        ("pipeline",),
        ("AutoModelForCausalLM",),
        ("AutoModelForSeq2SeqLM",),
        ("AutoTokenizer",),
    },
    "ollama": {("chat",), ("generate",),},
    "vllm": {("LLM",), ("SamplingParams",), ("EngineArgs",),},
    "llama_cpp": {("Llama",),},
    "google_genai": {("GenerativeModel",),},
    "openrouter": {("route",),},
}

MODEL_CONSTRUCTORS = {
    "transformers": {"AutoModelForCausalLM", "AutoModelForSeq2SeqLM"},
    "vllm": {"LLM"},
    "llama_cpp": {"Llama"},
    "google_genai": {"GenerativeModel"},
}

GENERATE_METHOD_NAMES = {
  "generate", "chat", "complete", "completions", "predict",
  "invoke", "run", "respond", "call", "create_chat_completion"
}

CALL_TAILS.update({
  "langchain": {("ChatOpenAI",), ("LLMChain",), ("PromptTemplate",)},
  "llama_index": {("LLMPredictor",), ("ServiceContext",)},
  "groq": {("Groq",), ("Client",), ("chat",), ("completions",)},
  "mistralai": {("Mistral",), ("Client",), ("chat",)},
  "cohere": {("Client",), ("generate",), ("chat",)},
})

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def rate_limit_guard(resp):
    try:
        remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
        reset     = int(resp.headers.get("X-RateLimit-Reset", "0"))
    except Exception:
        return
    if remaining <= 0 and reset > 0:
        now = int(time.time())
        wait = max(0, reset - now) + 5
        print(f"⏳ Limite atteinte. Pause {wait}s jusqu’à la fenêtre suivante…")
        time.sleep(wait)

def iter_attr_chain(node: ast.AST) -> List[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    return parts

from dataclasses import dataclass, field

@dataclass
class DetectionResult:
    provider: str
    lines: Set[int] = field(default_factory=set)
    evidence: List[str] = field(default_factory=list)

class LLMDetector(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()
        self.module_aliases: Dict[str, str] = {}
        self.known_vars: Dict[str, str] = {}
        self.results: Dict[str, DetectionResult] = {}

    def _mark(self, provider: str, node: ast.AST, msg: str):
        if provider not in self.results:
            self.results[provider] = DetectionResult(provider=provider)
        lineno = getattr(node, "lineno", None)
        if lineno is not None:
            self.results[provider].lines.add(lineno)
            if 1 <= lineno <= len(self.lines):
                snippet = self.lines[lineno-1].strip()
                self.results[provider].evidence.append(snippet if len(snippet) <= 200 else snippet[:200])

    def _provider_for_module(self, modname: str) -> Optional[str]:
        if not modname:
            return None
        for prov, modules in PROVIDER_MODULES.items():
            for m in modules:
                if modname == m or modname.startswith(m + "."):
                    return prov
        return None

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            mod = alias.name
            asname = alias.asname or alias.name.split(".")[0]
            prov = self._provider_for_module(mod)
            if prov:
                self.module_aliases[asname] = prov
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        prov = self._provider_for_module(mod)
        for alias in node.names:
            asname = alias.asname or alias.name
            if prov:
                self.module_aliases[asname] = prov
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        try:
            value = node.value
            provider = None
            constructor_name = None
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name):
                    name = func.id
                    prov = self.module_aliases.get(name) or self._provider_for_module(name)
                    if prov:
                        provider = prov
                        constructor_name = name
                elif isinstance(func, ast.Attribute):
                    chain = iter_attr_chain(func)
                    if chain:
                        base = chain[0]
                        prov = self.module_aliases.get(base) or self._provider_for_module(base)
                        if prov:
                            provider = prov
                            constructor_name = chain[-1]
            if provider and constructor_name:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.known_vars[target.id] = provider
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        try:
            func = node.func
            if isinstance(func, ast.Attribute):
                chain = iter_attr_chain(func)
                if chain:
                    base = chain[0]
                    tail = tuple(chain[1:]) if len(chain) > 1 else ()
                    prov = self.module_aliases.get(base) or self._provider_for_module(base)
                    if prov:
                        required_tails = CALL_TAILS.get(prov, set())
                        for sig in required_tails:
                            if tail[-len(sig):] == sig:
                                self._mark(prov, node, f"call:{'.'.join(chain)}")
                                break
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                var = func.value.id
                meth = func.attr
                prov = self.known_vars.get(var)
                if prov and (meth in {"generate","chat","complete","completions","predict"} or (meth,) in CALL_TAILS.get(prov, set())):
                    self._mark(prov, node, f"{var}.{meth}(...)")
            if isinstance(func, ast.Name):
                name = func.id
                prov = self.module_aliases.get(name) or self._provider_for_module(name)
                if not prov and name == "pipeline" and "transformers" in self.module_aliases.values():
                    prov = "transformers"
                if prov:
                    tails = CALL_TAILS.get(prov, set())
                    if (name,) in tails:
                        self._mark(prov, node, f"{name}(...)")
        except Exception:
            pass
        self.generic_visit(node)

def detect_llm_usages(source: str) -> Dict[str, DetectionResult]:
    try:
        tree = ast.parse(source)
    except Exception:
        return {}
    det = LLMDetector(source)
    det.visit(tree)
    return det.results

def download_repo_zip(owner: str, repo: str, sha: Optional[str], session: requests.Session, headers: Dict[str, str]) -> Optional[bytes]:
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    if sha:
        url += f"/{sha}"
    resp = session.get(url, headers=headers, timeout=180)
    if resp.status_code != 200:
        print(f"✖ Échec {resp.status_code} pour {owner}/{repo}{'@'+sha[:7] if sha else ''}")
        rate_limit_guard(resp)
        return None
    return resp.content

# ------------------------- Input/Output Configuration -------------------------


INPUT_CSV_PATH = "/Users/bramss/Documents/ETS/PhD/Code_Smell_LLM/Code_Smell_LLM/Prevalence/Dataset/merged_repos.csv"
OUTPUT_DIR = "/Users/bramss/Desktop/LLM_Files_Extracted"
OUTPUT_SUMMARY_CSV = "/Users/bramss/Desktop/llm_extraction_summary.csv"
GITHUB_TOKEN  = ""    


# Pause settings to avoid rate limits
PAUSE_EVERY_N_REPOS = 80
PAUSE_DURATION_SECONDS = 60

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=INPUT_CSV_PATH, help="CSV with columns owner,repo,commit_sha (optional)")
    ap.add_argument("--out", default=OUTPUT_DIR, help="Output directory for extracted LLM files (per repo subfolders)")
    ap.add_argument("--summary", default=OUTPUT_SUMMARY_CSV, help="CSV path for summary of detected files")
    ap.add_argument("--token", default=GITHUB_TOKEN, help="GitHub token")
    ap.add_argument("--pause-every", type=int, default=PAUSE_EVERY_N_REPOS, help="Pause after N repos to avoid rate limits")
    ap.add_argument("--pause-seconds", type=int, default=PAUSE_DURATION_SECONDS, help="Pause duration in seconds")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary).expanduser()
    token = args.token.strip()

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    print(f"🔎 {len(df)} dépôts à traiter depuis {csv_path}")

    session = requests.Session()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    rows = []
    processed = 0

    for idx, row in df.iterrows():
        owner = row["owner"].strip()
        repo = row["repo"].strip()
        sha = (row.get("commit_sha", "") or "").strip() or None

        tag = f"{owner}/{repo}" + (f"@{sha[:7]}" if sha else "")
        print(f"  ({idx+1}/{len(df)}) Traitement {tag}")

        content = download_repo_zip(owner, repo, sha, session, headers)
        if content is None:
            continue

        repo_dir = out_dir / f"{safe_filename(owner)}__{safe_filename(repo)}"
        repo_dir.mkdir(parents=True, exist_ok=True)

        per_repo_matches: List[Dict[str, object]] = []
        llm_files = 0

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for file_name in z.namelist():
                    if not file_name.endswith(".py"):
                        continue
                    try:
                        source = z.read(file_name).decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    detections = detect_llm_usages(source)
                    if not detections:
                        continue

                    parts = file_name.split("/", 1)
                    rel = parts[1] if len(parts) > 1 else parts[0]
                    out_path = repo_dir / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(source)

                    for prov, res in detections.items():
                        if not res.lines:
                            continue
                        llm_files += 1
                        per_repo_matches.append({
                            "owner": owner,
                            "repo": repo,
                            "commit_sha": sha or "",
                            "file": rel,
                            "provider": prov,
                            "lines_detected": sorted(list(res.lines)),
                            "evidence_snippets": res.evidence[:10],
                            "output_dir": str(repo_dir),
                        })
        except zipfile.BadZipFile:
            print(f"⚠️ ZIP corrompu pour {tag}")
            continue

        meta_path = repo_dir / "_llm_detection_meta.json"
        with open(meta_path, "w", encoding="utf-8") as jf:
            json.dump(per_repo_matches, jf, indent=2, ensure_ascii=False)

        rows.extend(per_repo_matches)
        print(f"✔ {llm_files} fichiers LLM extraits → {repo_dir}")

        processed += 1
        if processed % args.pause_every == 0:
            print(f"⏸ Pause {args.pause_seconds}s après {processed} dépôts…")
            time.sleep(args.pause_seconds)

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Extraction terminée. Résumé: {summary_csv}")

if __name__ == "__main__":
    main()