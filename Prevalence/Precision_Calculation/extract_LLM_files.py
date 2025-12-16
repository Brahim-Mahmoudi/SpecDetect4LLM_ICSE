import os, re, io, zipfile, json, requests
import pandas as pd
from pathlib import Path

# === CONFIG ===
CSV_FILE = "/Users/bramss/Documents/ETS/PhD/Code_Smell_LLM/Code_Smell_LLM/Prevalence/Dataset/merged_repos.csv"
OUTPUT_DIR = "/Users/bramss/Desktop/LLM_Files_Extracted"
SUMMARY_CSV = "/Users/bramss/Desktop/llm_extraction_summary.csv"
GITHUB_TOKEN  = ""   

PAUSE_EVERY = 50
PAUSE_SECONDS = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === PATTERNS LLM ===
LLM_PATTERNS = [
    r"openai", r"anthropic", r"ollama", r"vllm",
    r"transformers", r"AutoModel", r"pipeline\s*\(",
    r"ChatCompletion", r"Completion\.create", r"messages\.create",
    r"generate\s*\(", r"llm\s*=", r"prompt", r"temperature\s*="
]
LLM_REGEX = re.compile("|".join(LLM_PATTERNS), re.IGNORECASE)

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
        import time
        now = int(time.time())
        wait = max(0, reset - now) + 5
        print(f" Limite atteinte. Pause {wait}s jusqu’à la fenêtre suivante…")
        time.sleep(wait)

# === GITHUB SESSION ===
session = requests.Session()
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

# === ANALYSE ===
df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
summary = []
processed = 0

for idx, row in df.iterrows():
    owner = row["owner"]
    repo = row["repo"]
    commit_sha = row.get("commit_sha", "").strip() or None
    tag = f"{owner}/{repo}" + (f"@{commit_sha[:7]}" if commit_sha else "")
    print(f"\n  ({idx+1}/{len(df)}) Téléchargement et extraction de {tag}")

    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    if commit_sha:
        zip_url += f"/{commit_sha}"

    try:
        resp = session.get(zip_url, headers=headers, timeout=180)
        if resp.status_code != 200:
            print(f" Échec {resp.status_code} pour {tag}")
            rate_limit_guard(resp)
            continue
    except Exception as e:
        print(f" Erreur réseau {tag}: {e}")
        continue

    repo_dir = os.path.join(OUTPUT_DIR, f"{safe_filename(owner)}__{safe_filename(repo)}")
    os.makedirs(repo_dir, exist_ok=True)

    llm_files = 0
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for file_name in z.namelist():
                if not file_name.endswith(".py"):
                    continue
                try:
                    source = z.read(file_name).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if LLM_REGEX.search(source):
                    rel_name = file_name.split("/", 1)[-1]
                    out_path = os.path.join(repo_dir, rel_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(source)
                    llm_files += 1
    except zipfile.BadZipFile:
        print(f" ZIP corrompu pour {tag}")
        continue

    summary.append({
        "owner": owner,
        "repo": repo,
        "commit_sha": commit_sha or "",
        "llm_files": llm_files,
        "output_dir": repo_dir
    })
    print(f"✔ {llm_files} fichiers LLM extraits dans {repo_dir}")

    processed += 1
    if processed % PAUSE_EVERY == 0:
        import time
        print(f" Pause {PAUSE_SECONDS}s après {processed} dépôts…")
        time.sleep(PAUSE_SECONDS)

# === EXPORT DU RÉSUMÉ ===
pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
print(f"\n Extraction terminée. Résumé: {SUMMARY_CSV}")
