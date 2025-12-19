import os
import re
import io
import time
import json
import ast
import zipfile
import requests
import pandas as pd

from datetime import datetime


# === CONFIG (edit paths as needed) ===
CSV_FILE     = "pplication.csv"            # path to your CSV
OUTPUT_DIR   = "Extracted_Repo_ICSE"          # where to extract repositories
ZIP_DIR      = "repo_zips"      # where to store .zip files
ANALYSIS_DIR = "repo_analyses"  # per-repo JSON metrics

SAVE_ZIP      = True   # True => store zip files on disk
EXTRACT_REPO  = True   # True => extract zip on disk
DO_ANALYSIS   = True   # True => compute AST-based metrics
SKIP_EXISTING = True   # True => skip if extraction/analysis already exists
PAUSE_EVERY   = 100    # pause every N repos
PAUSE_SECONDS = 60

TOKEN_ENV_VAR = "GITHUB_TOKEN"   # environment variable containing the GitHub token
GITHUB_TOKEN = os.getenv(TOKEN_ENV_VAR, "").strip()


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# === UTILS ===
def parse_owner_repo_from_url(url: str):
    """
    Accepted formats:
      - https://github.com/owner/repo
      - http://github.com/owner/repo
      - git@github.com:owner/repo.git
      - owner/repo
    Returns (owner, repo) or (None, None).
    """
    if not isinstance(url, str) or not url.strip():
        return None, None
    s = url.strip()

    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", s):
        owner, repo = s.split("/", 1)
        return owner, repo.replace(".git", "")

    m = re.search(r"github\.com[:/]+(?P<owner>[^/]+)/(?P<repo>[^/#?]+)", s, flags=re.I)
    if m:
        owner = m.group("owner")
        repo = m.group("repo").replace(".git", "")
        return owner, repo

    return None, None


def unique_owner_repo_pairs(df: pd.DataFrame):
    """
    Typical CSV columns:
      - APP (e.g., owner/repo)
      - url (e.g., https://github.com/owner/repo)
      - commit id (optional)
    Returns list of (owner, repo, commit_sha_or_None).
    """
    triples = []

    if "APP" in df.columns:
        for app, sha in zip(df["APP"], df.get("commit id", [None] * len(df))):
            owner, repo = parse_owner_repo_from_url(app)
            if owner and repo:
                sha_val = str(sha).strip() if isinstance(sha, str) and sha.strip() else None
                triples.append((owner.strip(), repo.strip(), sha_val))

    if "url" in df.columns:
        commit_series = df.get("commit id", [None] * len(df))
        for url, sha in zip(df["url"], commit_series):
            owner, repo = parse_owner_repo_from_url(url)
            if owner and repo:
                sha_val = str(sha).strip() if isinstance(sha, str) and sha.strip() else None
                triples.append((owner.strip(), repo.strip(), sha_val))

    alt_full_cols = ["Repository", "Repo", "full_name", "Full Name", "Dépôt", "Nom complet"]
    for col in alt_full_cols:
        if col in df.columns:
            for val, sha in zip(df[col], df.get("commit id", [None] * len(df))):
                owner, repo = parse_owner_repo_from_url(val)
                if owner and repo:
                    sha_val = str(sha).strip() if isinstance(sha, str) and sha.strip() else None
                    triples.append((owner.strip(), repo.strip(), sha_val))

    return sorted(set(triples))


def rate_limit_guard(resp):
    try:
        remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
    except Exception:
        return

    if remaining <= 0 and reset > 0:
        now = int(time.time())
        wait = max(0, reset - now) + 5
        print(f"Rate limit reached. Sleeping {wait}s until the next window.")
        time.sleep(wait)


def safe_filename(s: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def compute_max_depth(node, depth=0):
    m = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            m = max(m, compute_max_depth(child, depth + 1))
        else:
            m = max(m, compute_max_depth(child, depth))
    return m


def analyze_python_file(path: str, rel_tag: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=rel_tag)
    except Exception:
        return None

    stats = {
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
                base_name = (
                    base.id if isinstance(base, ast.Name)
                    else base.attr if isinstance(base, ast.Attribute)
                    else ""
                )
                if "pipeline" in str(base_name).lower():
                    is_pipeline = True
            if is_pipeline:
                stats["custom_pipeline_classes"] += 1

        elif isinstance(node, ast.Assign):
            stats["num_assignments"] += 1

        elif isinstance(node, ast.Call):
            func = node.func
            func_name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else ""
            )
            fl = str(func_name).lower()
            if "pipeline" in fl:
                stats["pipeline_patterns"] += 1
            if func_name in {"getattr", "setattr", "exec", "eval", "__import__"}:
                stats["has_dynamic_construct"] = True

    depths = [compute_max_depth(n, depth=1) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    stats["max_function_depth"] = max(depths) if depths else 0
    return stats


def extract_zip_to(zip_path: str, dest_root: str, owner: str, repo: str):
    final_dir = os.path.join(dest_root, f"{safe_filename(owner)}__{safe_filename(repo)}")
    if SKIP_EXISTING and os.path.exists(final_dir) and os.listdir(final_dir):
        return final_dir

    tmp_dir = final_dir + "__partial"
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)

    roots = set(n.split("/")[0] for n in z.namelist() if "/" in n)
    os.makedirs(final_dir, exist_ok=True)

    if len(roots) == 1:
        root = list(roots)[0]
        src_root = os.path.join(tmp_dir, root)
        for dirpath, _, filenames in os.walk(src_root):
            rel = os.path.relpath(dirpath, src_root)
            tgt = os.path.join(final_dir, rel) if rel != "." else final_dir
            os.makedirs(tgt, exist_ok=True)
            for fn in filenames:
                sp = os.path.join(dirpath, fn)
                dp = os.path.join(tgt, fn)
                os.replace(sp, dp)
    else:
        for dirpath, _, filenames in os.walk(tmp_dir):
            rel = os.path.relpath(dirpath, tmp_dir)
            tgt = os.path.join(final_dir, rel) if rel != "." else final_dir
            os.makedirs(tgt, exist_ok=True)
            for fn in filenames:
                os.replace(os.path.join(dirpath, fn), os.path.join(tgt, fn))

    try:
        for dirpath, dirnames, filenames in os.walk(tmp_dir, topdown=False):
            for fn in filenames:
                try:
                    os.remove(os.path.join(dirpath, fn))
                except Exception:
                    pass
            for d in dirnames:
                try:
                    os.rmdir(os.path.join(dirpath, d))
                except Exception:
                    pass
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return final_dir


# === LOAD CSV ===
df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
pairs = unique_owner_repo_pairs(df)

print(f"Detected {len(pairs)} unique repositories in {CSV_FILE}.")

session = requests.Session()
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"


processed = 0

for idx, (owner, repo, commit_sha) in enumerate(pairs, start=1):
    if commit_sha:
        zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{commit_sha}"
        tag = f"{owner}/{repo}@{commit_sha[:7]}"
    else:
        zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
        tag = f"{owner}/{repo}"

    print(f"\n({idx}/{len(pairs)}) Downloading {tag}")

    final_dir = os.path.join(OUTPUT_DIR, f"{safe_filename(owner)}__{safe_filename(repo)}")
    analysis_path = os.path.join(ANALYSIS_DIR, f"{owner}-{repo}_analysis.json")

    if SKIP_EXISTING and os.path.exists(final_dir) and os.path.exists(analysis_path):
        print("Already extracted and analyzed. Skipping.")
        continue

    try:
        resp = session.get(zip_url, headers=headers, stream=True, timeout=120)
    except Exception as e:
        print(f"Network error for {tag}: {e}")
        continue

    if resp.status_code != 200:
        code = resp.status_code
        msg = "not found (404)" if code == 404 else "forbidden (403)" if code == 403 else f"failed (HTTP {code})"
        print(f"Failed for {tag}: {msg}")
        rate_limit_guard(resp)
        continue

    zip_path = None
    data = None

    if SAVE_ZIP:
        zip_name = f"{safe_filename(owner)}__{safe_filename(repo)}"
        if commit_sha:
            zip_name += f"__{commit_sha[:12]}"
        zip_name += ".zip"
        zip_path = os.path.join(ZIP_DIR, zip_name)

        try:
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=2**20):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Could not write zip for {tag}: {e}")
            continue
    else:
        data = resp.content

    extracted_dir = None
    try:
        if EXTRACT_REPO:
            if SAVE_ZIP and zip_path:
                extracted_dir = extract_zip_to(zip_path, OUTPUT_DIR, owner, repo)
            else:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    tmp_zip = os.path.join(ZIP_DIR, f"__mem_{safe_filename(owner)}__{safe_filename(repo)}.zip")
                    with open(tmp_zip, "wb") as f:
                        f.write(data)
                    extracted_dir = extract_zip_to(tmp_zip, OUTPUT_DIR, owner, repo)
                    try:
                        os.remove(tmp_zip)
                    except Exception:
                        pass
        else:
            extracted_dir = None
    except zipfile.BadZipFile:
        print(f"Corrupted ZIP for {tag}")
        continue
    except Exception as e:
        print(f"Extraction error for {tag}: {e}")
        continue

    if DO_ANALYSIS:
        results = []
        try:
            if extracted_dir and os.path.isdir(extracted_dir):
                for root, _, files in os.walk(extracted_dir):
                    for fn in files:
                        if fn.endswith(".py"):
                            full = os.path.join(root, fn)
                            rel = os.path.relpath(full, extracted_dir)
                            stat = analyze_python_file(full, f"{owner}/{repo}/{rel}")
                            if stat:
                                results.append(stat)
            else:
                if SAVE_ZIP and zip_path:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        for file_name in z.namelist():
                            if file_name.endswith(".py"):
                                try:
                                    source = z.read(file_name).decode("utf-8", errors="ignore")
                                    tree = ast.parse(source, filename=file_name)
                                except Exception:
                                    continue

                                stats = {
                                    "file": f"{owner}/{repo}/{file_name}",
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
                                        name_lower = node.name.lower()
                                        is_pipeline = "pipeline" in name_lower
                                        for base in node.bases:
                                            base_name = (
                                                base.id if isinstance(base, ast.Name)
                                                else base.attr if isinstance(base, ast.Attribute)
                                                else ""
                                            )
                                            if "pipeline" in str(base_name).lower():
                                                is_pipeline = True
                                        if is_pipeline:
                                            stats["custom_pipeline_classes"] += 1
                                    elif isinstance(node, ast.Assign):
                                        stats["num_assignments"] += 1
                                    elif isinstance(node, ast.Call):
                                        func = node.func
                                        func_name = (
                                            func.id if isinstance(func, ast.Name)
                                            else func.attr if isinstance(func, ast.Attribute)
                                            else ""
                                        )
                                        if "pipeline" in str(func_name).lower():
                                            stats["pipeline_patterns"] += 1
                                        if func_name in {"getattr", "setattr", "exec", "eval", "__import__"}:
                                            stats["has_dynamic_construct"] = True

                                depths = [compute_max_depth(n, depth=1) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                                stats["max_function_depth"] = max(depths) if depths else 0
                                results.append(stats)
                else:
                    results = []
        except Exception as e:
            print(f"Partial/failed analysis for {tag}: {e}")

        try:
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Could not write analysis JSON for {tag}: {e}")

        print(f"{tag}: analyzed {len(results)} Python files.")

    processed += 1
    if processed % PAUSE_EVERY == 0:
        print(f"Pausing {PAUSE_SECONDS}s after {processed} repos.")
        time.sleep(PAUSE_SECONDS)
