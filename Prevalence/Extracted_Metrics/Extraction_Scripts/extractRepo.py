

import os, re, io, time, json, ast, zipfile, requests
import pandas as pd
from datetime import datetime

import os, re, io, time, json, ast, zipfile, requests
import pandas as pd
from datetime import datetime

# === CONFIG ===
CSV_FILE     = "/Users/bramss/Downloads/application.csv"      # chemin vers ton CSV
OUTPUT_DIR   = "/Users/bramss/Desktop/Extracted_Repo_ICSE"    # où extraire les dépôts
ZIP_DIR      = "/Users/bramss/Desktop/Extracted_Repo_ICSE/repo_zips"     # où sauvegarder les .zip
ANALYSIS_DIR = "/Users/bramss/Desktop/Extracted_Repo_ICSE/repo_analyses" # JSON de métriques par dépôt

SAVE_ZIP        = True   # True => sauvegarder les zip
EXTRACT_REPO    = True   # True => extraire le zip sur disque
DO_ANALYSIS     = True   # True => calculer des métriques AST
SKIP_EXISTING   = True   # True => saute si déjà extrait/analyse présent
PAUSE_EVERY     = 100    # pause toutes les 100 requêtes
PAUSE_SECONDS   = 60

GITHUB_TOKEN = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# === UTILS ===
def parse_owner_repo_from_url(url: str):
    """
    Accepte:
      - https://github.com/owner/repo
      - http://github.com/owner/repo
      - git@github.com:owner/repo.git
      - owner/repo
    Renvoie (owner, repo) ou (None, None).
    """
    if not isinstance(url, str) or not url.strip():
        return None, None
    s = url.strip()

    # owner/repo direct
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", s):
        owner, repo = s.split("/", 1)
        return owner, repo.replace(".git", "")

    # HTTPS / SSH
    m = re.search(r"github\.com[:/]+(?P<owner>[^/]+)/(?P<repo>[^/#?]+)", s, flags=re.I)
    if m:
        owner = m.group("owner")
        repo  = m.group("repo").replace(".git", "")
        return owner, repo

    return None, None

def unique_owner_repo_pairs(df: pd.DataFrame):
    """
    Pour ton CSV typique avec colonnes:
      - APP (ex: owner/repo)
      - url (ex: https://github.com/owner/repo)
      - commit id (optionnel)
    Renvoie une liste de tuples (owner, repo, commit_sha_ou_None).
    """
    triples = []

    # 1) APP = "owner/repo"
    if "APP" in df.columns:
        for app, sha in zip(df["APP"], df.get("commit id", [None]*len(df))):
            owner, repo = parse_owner_repo_from_url(app)
            if owner and repo:
                triples.append((owner.strip(), repo.strip(), str(sha).strip() if isinstance(sha, str) and sha.strip() else None))

    # 2) url
    if "url" in df.columns:
        # si commit id colonne existe, l'utiliser
        commit_series = df.get("commit id", [None]*len(df))
        for url, sha in zip(df["url"], commit_series):
            owner, repo = parse_owner_repo_from_url(url)
            if owner and repo:
                triples.append((owner.strip(), repo.strip(), str(sha).strip() if isinstance(sha, str) and sha.strip() else None))

    # 3) fallback si noms alternatifs
    alt_full_cols = ["Repository", "Repo", "full_name", "Full Name", "Dépôt", "Nom complet"]
    for col in alt_full_cols:
        if col in df.columns:
            for val, sha in zip(df[col], df.get("commit id", [None]*len(df))):
                owner, repo = parse_owner_repo_from_url(val)
                if owner and repo:
                    triples.append((owner.strip(), repo.strip(), str(sha).strip() if isinstance(sha, str) and sha.strip() else None))

    # Uniques par (owner, repo, sha)
    uniq = sorted(set(triples))
    return uniq

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

def safe_filename(s: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def compute_max_depth(node, depth=0):
    m = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            m = max(m, compute_max_depth(child, depth+1))
        else:
            m = max(m, compute_max_depth(child, depth))
    return m

def analyze_python_file(path: str, rel_tag: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=rel_tag)
    except Exception:
        return None  # non analysable

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
        for dirpath, dirnames, filenames in os.walk(src_root):
            rel = os.path.relpath(dirpath, src_root)
            tgt = os.path.join(final_dir, rel) if rel != "." else final_dir
            os.makedirs(tgt, exist_ok=True)
            for f in filenames:
                sp = os.path.join(dirpath, f)
                dp = os.path.join(tgt, f)
                os.replace(sp, dp)
    else:
        for dirpath, dirnames, filenames in os.walk(tmp_dir):
            rel = os.path.relpath(dirpath, tmp_dir)
            tgt = os.path.join(final_dir, rel) if rel != "." else final_dir
            os.makedirs(tgt, exist_ok=True)
            for f in filenames:
                os.replace(os.path.join(dirpath, f), os.path.join(tgt, f))

    # Nettoyage
    try:
        for dirpath, dirnames, filenames in os.walk(tmp_dir, topdown=False):
            for f in filenames:
                try:
                    os.remove(os.path.join(dirpath, f))
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

# === CHARGEMENT CSV ===
# Astuce: keep_default_na=False pour éviter que "None"/"NA" soient convertis en NaN
df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
pairs = unique_owner_repo_pairs(df)

print(f"🔎 {len(pairs)} dépôts uniques détectés dans {CSV_FILE}.")

session = requests.Session()
headers = {
    "Accept": "application/vnd.github+json",
}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

processed = 0

for idx, (owner, repo, commit_sha) in enumerate(pairs, start=1):
    # Si commit id fourni, on le cible explicitement
    if commit_sha:
        zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{commit_sha}"
        tag = f"{owner}/{repo}@{commit_sha[:7]}"
    else:
        zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
        tag = f"{owner}/{repo}"

    print(f"\n➡️  ({idx}/{len(pairs)}) Téléchargement de {tag}…")

    final_dir = os.path.join(OUTPUT_DIR, f"{safe_filename(owner)}__{safe_filename(repo)}")
    analysis_path = os.path.join(ANALYSIS_DIR, f"{owner}-{repo}_analysis.json")
    if SKIP_EXISTING and os.path.exists(final_dir) and os.path.exists(analysis_path):
        print(f"↩️  Déjà présent (extraction + analyse). On saute.")
        continue

    try:
        resp = session.get(zip_url, headers=headers, stream=True, timeout=120)
    except Exception as e:
        print(f"✖ Erreur réseau pour {tag}: {e}")
        continue

    if resp.status_code != 200:
        code = resp.status_code
        msg = "introuvable (404)" if code == 404 else "accès refusé (403)" if code == 403 else f"échec (code {code})"
        print(f"✖ Dépôt {tag} {msg}")
        rate_limit_guard(resp)
        continue

    # Enregistre le zip si demandé
    zip_path = None
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
            print(f"✖ Impossible d’écrire le zip pour {tag}: {e}")
            continue
    else:
        data = resp.content
        zip_path = None

    # Extraire si demandé
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
        print(f"✖ Archive ZIP corrompue pour {tag}")
        continue
    except Exception as e:
        print(f"✖ Erreur à l’extraction pour {tag}: {e}")
        continue

    # Analyse AST
    if DO_ANALYSIS:
        results = []
        try:
            if extracted_dir and os.path.isdir(extracted_dir):
                for root, _, files in os.walk(extracted_dir):
                    for fn in files:
                        if fn.endswith(".py"):
                            full = os.path.join(root, fn)
                            rel  = os.path.relpath(full, extracted_dir)
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
                                    "loc": len(source.splitlines())
                                }
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.FunctionDef):
                                        stats["num_functions"] += 1
                                    elif isinstance(node, ast.ClassDef):
                                        stats["num_classes"] += 1
                                        name_lower = node.name.lower()
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
            print(f"⚠️  Analyse partielle/échouée pour {tag}: {e}")
            results = results if 'results' in locals() else []

        try:
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"✖ Impossible d’écrire l’analyse JSON pour {tag}: {e}")

        print(f"✔ {tag}: {len(results)} fichiers Python analysés.")

    processed += 1
    if processed % PAUSE_EVERY == 0:
        print(f"⏸ Pause {PAUSE_SECONDS}s après {processed} dépôts traités…")
        time.sleep(PAUSE_SECONDS)
