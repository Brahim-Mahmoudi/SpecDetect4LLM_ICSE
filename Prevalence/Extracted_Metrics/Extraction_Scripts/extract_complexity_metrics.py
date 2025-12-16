import os, io, re, ast, json, time, zipfile, requests, pandas as pd
from datetime import datetime

# === CONFIGURATION ===
CSV_FILE      = "/Users/bramss/Documents/ETS/PhD/Code_Smell_LLM/Code_Smell_LLM/Prevalence/Dataset/merged_repos.csv"       
OUTPUT_DIR    = "/Users/bramss/Desktop/repo_analyses"          # dossier de sortie des JSON
GITHUB_TOKEN  = "XXX"    # ton token GitHub personnel

PAUSE_EVERY   = 100      # pause toutes les N requêtes
PAUSE_SECONDS = 60       # durée de la pause

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === UTILS ===
def safe_filename(s: str):
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
        print(f" Limite atteinte. Pause {wait}s jusqu’à la fenêtre suivante…")
        time.sleep(wait)

def compute_max_depth(node, depth=0):
    m = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            m = max(m, compute_max_depth(child, depth+1))
        else:
            m = max(m, compute_max_depth(child, depth))
    return m

def analyze_python_source(source: str, rel_tag: str):
    """Analyse AST d’un fichier source Python brut"""
    try:
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
        "loc": len(source.splitlines())
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats["num_functions"] += 1
        elif isinstance(node, ast.ClassDef):
            stats["num_classes"] += 1
            name_lower = getattr(node, "name", "").lower()
            is_pipeline = "pipeline" in name_lower
            for base in node.bases:
                base_name = (base.id if isinstance(base, ast.Name)
                             else base.attr if isinstance(base, ast.Attribute)
                             else "")
                if "pipeline" in str(base_name).lower():
                    is_pipeline = True
            if is_pipeline:
                stats["custom_pipeline_classes"] += 1
        elif isinstance(node, ast.Assign):
            stats["num_assignments"] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            func_name = (func.id if isinstance(func, ast.Name)
                         else func.attr if isinstance(func, ast.Attribute)
                         else "")
            fl = str(func_name).lower()
            if "pipeline" in fl:
                stats["pipeline_patterns"] += 1
            if func_name in {"getattr", "setattr", "exec", "eval", "__import__"}:
                stats["has_dynamic_construct"] = True

    depths = [compute_max_depth(n, depth=1) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    stats["max_function_depth"] = max(depths) if depths else 0
    return stats

# === CHARGEMENT CSV ===
df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
print(f" {len(df)} dépôts à traiter")

session = requests.Session()
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

processed = 0

for idx, row in df.iterrows():
    owner = row["owner"]
    repo = row["repo"]
    commit_sha = row.get("commit_sha", "").strip() or None

    tag = f"{owner}/{repo}" + (f"@{commit_sha[:7]}" if commit_sha else "")
    print(f"\n  ({idx+1}/{len(df)}) Analyse de {tag}...")

    json_path = os.path.join(OUTPUT_DIR, f"{safe_filename(owner)}__{safe_filename(repo)}.json")
    if os.path.exists(json_path):
        print("↩  JSON déjà présent, on saute.")
        continue

    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    if commit_sha:
        zip_url += f"/{commit_sha}"

    try:
        resp = session.get(zip_url, headers=headers, timeout=180)
        if resp.status_code != 200:
            print(f"✖ Échec ({resp.status_code}) pour {tag}")
            rate_limit_guard(resp)
            continue
    except Exception as e:
        print(f" Erreur réseau pour {tag}: {e}")
        continue

    # Lecture du ZIP directement en mémoire
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for file_name in z.namelist():
                if file_name.endswith(".py"):
                    try:
                        source = z.read(file_name).decode("utf-8", errors="ignore")
                        stats = analyze_python_source(source, f"{owner}/{repo}/{file_name}")
                        if stats:
                            results.append(stats)
                    except Exception:
                        continue
    except zipfile.BadZipFile:
        print(f" Archive corrompue pour {tag}")
        continue

    # Sauvegarde du JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f" {tag}: {len(results)} fichiers analysés → {json_path}")

    processed += 1
    if processed % PAUSE_EVERY == 0:
        print(f"⏸ Pause de {PAUSE_SECONDS}s après {processed} dépôts...")
        time.sleep(PAUSE_SECONDS)

print(f"\n Terminé. {processed} dépôts traités. Résultats dans {OUTPUT_DIR}")
