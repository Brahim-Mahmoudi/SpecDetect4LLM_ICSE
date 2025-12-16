#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregated analysis of SpecDetect4AI results — visuals + HTML report.

Inputs (folder):
  - Multiple JSON files (one per project), each shaped like:
      { filepath: { rule: [messages…] }, ... }

Outputs (under --out-dir):
  - CSV: rule_counts.csv, project_rule_counts.csv, file_alerts.csv,
         rule_projects_affected.csv
  - PNG: top_rules.png, pareto_rules.png, top_projects.png,
         heatmap_rules_projects.png, rules_coverage_scatter.png, per_rule_boxplot.png,
         projects_affected.png
  - summary.md
  - report.html  (self-contained HTML; if --inline-images, embeds images as base64)

Dependencies:
  - pandas, matplotlib
    -> pip install pandas matplotlib
"""

import argparse
import base64
import io
import json
from pathlib import Path
from textwrap import wrap
import datetime
import re

import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional  


# ---------- Rule renaming (R25–R29) ----------

_RULE_RENAMES = {
    "R25": "LLM Temperature Not Explicitly Set(TNES)",
    "R26": "No Model Version Pinning (NMVP)",
    "R27": "No System Message (NSM)",
    "R28": "Unbounded Max Metrics (UMM)",
    "R29": "No Structured Output (NSO)",
}

# Matches: "R26", "rule R26", "R26: something", "Rule: R26", etc.
_RXX_EXTRACT = re.compile(r"\bR(?P<num>25|26|27|28|29)\b", flags=re.IGNORECASE)


def normalize_rule_name(rule_raw: str) -> str:
    """
    Normalize a rule identifier to the requested English display name.
    - If the rule looks like R25..R29 (with or without prefixes/suffixes), map to the long label.
    - Otherwise, return the original rule string.
    """
    if not isinstance(rule_raw, str):
        return str(rule_raw)
    m = _RXX_EXTRACT.search(rule_raw)
    if m:
        key = f"R{m.group('num')}"
        return _RULE_RENAMES.get(key, rule_raw)
    return rule_raw


# ---------- Loading & transformation ----------

def load_reports(input_dirs: List[Path], source_labels: Optional[List[str]]):
    """
    Load all JSON files from multiple folders (ignores batch_summary.json).
    Returns a list of row dicts: project, file_path, rule, n_alerts, messages, source.
    """
    rows = []
    if source_labels is None:
        source_labels = [p.name for p in input_dirs]
    if len(source_labels) != len(input_dirs):
        raise ValueError("--source-labels must have the same length as --input-dir")

    for src_dir, src_label in zip(input_dirs, source_labels):
        for p in sorted(src_dir.glob("*.json")):
            if p.name == "batch_summary.json":
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not (isinstance(data, dict) and all(isinstance(v, dict) for v in data.values())):
                    continue
                project = p.stem
                for fpath, rules_map in data.items():
                    if not isinstance(rules_map, dict):
                        continue
                    for rule, messages in rules_map.items():
                        n = 0
                        if isinstance(messages, list):
                            n = len(messages)
                        elif messages is not None:
                            n = 1
                            messages = [str(messages)]
                        rows.append({
                            "source": src_label,
                            "project": project,
                            "file_path": fpath,
                            "rule": normalize_rule_name(rule),
                            "n_alerts": n,
                            "messages": messages if isinstance(messages, list) else [],
                        })
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
    return rows



def explode_to_rows(rows):
    cols = ["source", "project", "file_path", "rule", "n_alerts", "messages"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return df

def compute_summaries_by_source(df: pd.DataFrame):
    """Per-source aggregates."""
    if df.empty:
        return (
            pd.DataFrame(columns=["source","rule","total_alerts"]),
            pd.DataFrame(columns=["source","project","rule","alerts"]),
            pd.DataFrame(columns=["source","project","file_path","rule","n_alerts"]),
        )
    rule_counts_src = (
        df.groupby(["source","rule"])["n_alerts"].sum()
          .reset_index().rename(columns={"n_alerts":"total_alerts"})
          .sort_values(["source","total_alerts"], ascending=[True, False])
    )
    project_rule_src = (
        df.groupby(["source","project","rule"])["n_alerts"].sum()
          .reset_index().rename(columns={"n_alerts":"alerts"})
    )
    file_alerts_src = df[["source","project","file_path","rule","n_alerts"]].copy()
    return rule_counts_src, project_rule_src, file_alerts_src


def compute_rule_coverage_by_source(project_rule: pd.DataFrame) -> pd.DataFrame:
    """Distinct projects affected per rule per source."""
    if project_rule.empty:
        return pd.DataFrame(columns=["source","rule","projects_affected"])
    cov = (project_rule.groupby(["source","rule"])["project"]
           .nunique().reset_index().rename(columns={"project":"projects_affected"}))
    return cov.sort_values(["source","projects_affected"], ascending=[True, False])


def compute_summaries(df: pd.DataFrame):
    """Compute main aggregates + headline KPIs."""
    if df.empty:
        headline = {"projects": 0, "files_with_alerts": 0, "total_alerts": 0, "distinct_rules": 0}
        return (pd.Series(dtype=int), pd.DataFrame(), pd.DataFrame(), headline)

    rule_counts = df.groupby("rule")["n_alerts"].sum().sort_values(ascending=False)
    project_rule = df.groupby(["project", "rule"])["n_alerts"].sum().reset_index()
    file_alerts = df.groupby(["project", "file_path", "rule"])["n_alerts"].sum().reset_index()

    headline = {
        "projects": df["project"].nunique(),
        "files_with_alerts": df["file_path"].nunique(),
        "total_alerts": int(df["n_alerts"].sum()),
        "distinct_rules": df["rule"].nunique(),
    }
    return rule_counts, project_rule, file_alerts, headline


# ---------- NEW: coverage metric (distinct projects per rule) ----------

def compute_rule_coverage(project_rule: pd.DataFrame) -> pd.Series:
    """
    Returns a Series: index=rule, value=number of distinct projects affected.
    Each project is counted at most once per rule, regardless of alert count.
    """
    if project_rule.empty:
        return pd.Series(dtype=int)
    return project_rule.groupby("rule")["project"].nunique().sort_values(ascending=False)


# ---------- Plot helpers ----------

def _annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if h is None:
            continue
        ax.annotate(f"{int(h)}", (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")


def _annotate_hbars(ax):
    for p in ax.patches:
        w = p.get_width()
        if w is None:
            continue
        ax.annotate(f"{int(w)}", (w, p.get_y() + p.get_height()/2),
                    ha="left", va="center", fontsize=9, xytext=(3, 0), textcoords="offset points")


def _wrap_labels(labels, width=22):
    return ["\n".join(wrap(str(x), width=width)) for x in labels]


# ---------- Plots ----------

def plot_top_rules(rule_counts: pd.Series, out_path: Path, top_n: int = 15):
    if rule_counts.empty:
        return
    top = rule_counts.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(top)), 5))
    top.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {min(top_n, len(rule_counts))} Rules by Total Alerts")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Total Alerts")
    ax.set_xticklabels(_wrap_labels(top.index, 16), rotation=0)
    _annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pareto(rule_counts: pd.Series, out_path: Path, top_n: int = 20):
    if rule_counts.empty:
        return
    s = rule_counts.head(top_n)
    cum = s.cumsum() / s.sum() * 100.0
    fig, ax1 = plt.subplots(figsize=(max(8, 0.7 * len(s)), 5))
    s.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Total Alerts")
    ax1.set_xlabel("Rule")
    ax1.set_xticklabels(_wrap_labels(s.index, 16), rotation=0)
    _annotate_bars(ax1)
    ax2 = ax1.twinx()
    ax2.plot(range(len(cum)), cum.values, marker="o")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 110)
    ax2.grid(False)
    ax1.set_title("Pareto of Rule Frequencies")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_projects(project_rule: pd.DataFrame, out_path: Path, top_projects: int = 15):
    if project_rule.empty:
        return
    totals = project_rule.groupby("project")["n_alerts"].sum().sort_values(ascending=False).head(top_projects)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(totals))))
    totals.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Projects with Most Alerts (Top {len(totals)})")
    ax.set_xlabel("Total Alerts")
    ax.set_ylabel("Project")
    ax.set_yticklabels(_wrap_labels(totals.sort_values().index, 28))
    _annotate_hbars(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap_rules_projects(project_rule: pd.DataFrame, out_path: Path,
                                top_rules: int = 8, top_projects: int = 12):
    if project_rule.empty:
        return
    top_r = (project_rule.groupby("rule")["n_alerts"]
             .sum().sort_values(ascending=False).head(max(1, top_rules))).index.tolist()
    top_p = (project_rule.groupby("project")["n_alerts"]
             .sum().sort_values(ascending=False).head(max(1, top_projects))).index.tolist()
    df = project_rule.copy()
    df = df[df["rule"].isin(top_r) & df["project"].isin(top_p)]
    if df.empty:
        return
    mat = df.pivot_table(index="rule", columns="project", values="n_alerts", aggfunc="sum", fill_value=0)
    mat = mat.loc[top_r, top_p]
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * mat.shape[1]), max(5, 0.6 * mat.shape[0])))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(_wrap_labels(list(mat.columns), 18), rotation=90, va="top")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.set_title("Heatmap — Alerts by Rule × Project (restricted)")
    ax.set_xlabel("Project")
    ax.set_ylabel("Rule")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat.values[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Alerts")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rules_coverage_scatter(project_rule: pd.DataFrame, out_path: Path, top_rules: int = 30):
    if project_rule.empty:
        return
    totals = project_rule.groupby("rule")["n_alerts"].sum().sort_values(ascending=False)
    coverage = project_rule.groupby("rule")["project"].nunique()
    s = pd.concat([totals, coverage], axis=1)
    s.columns = ["total_alerts", "projects_affected"]
    s = s.sort_values("total_alerts", ascending=False).head(max(1, top_rules))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s["total_alerts"], s["projects_affected"])
    for rule, row in s.iterrows():
        ax.annotate(rule, (row["total_alerts"], row["projects_affected"]), fontsize=9, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_title("Rule Coverage: Frequency vs. Affected Projects")
    ax.set_xlabel("Total Alerts (rule)")
    ax.set_ylabel("Projects Affected (rule)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_rule_boxplot(project_rule: pd.DataFrame, out_path: Path, top_rules: int = 10):
    if project_rule.empty:
        return
    top_r = (project_rule.groupby("rule")["n_alerts"]
             .sum().sort_values(ascending=False).head(max(1, top_rules))).index.tolist()
    df = project_rule[project_rule["rule"].isin(top_r)]
    data, labels = [], []
    for r in top_r:
        vals = df[df["rule"] == r].groupby("project")["n_alerts"].sum().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(r)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5))
    ax.boxplot(data, labels=_wrap_labels(labels, 16), showfliers=False)
    ax.set_title("Per-Rule Distribution of Alerts Across Projects")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Alerts per Project")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------- NEW: bar chart for distinct projects affected per rule ----------

def plot_projects_affected_bar(project_rule: pd.DataFrame, out_path: Path, top_n: int = 15):
    if project_rule.empty:
        return
    cov = compute_rule_coverage(project_rule).head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(cov)), 5))
    cov.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {min(top_n, len(cov))} Rules by Projects Affected")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Projects Affected")
    ax.set_xticklabels(_wrap_labels(cov.index, 16), rotation=0)
    _annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------- Exports: text/CSV/HTML ----------


def save_csvs(out_dir: Path, rule_counts, project_rule, file_alerts,
              rule_coverage: pd.Series,
              by_source: bool = False,
              rule_counts_src: Optional[pd.DataFrame] = None,
              project_rule_src: Optional[pd.DataFrame] = None,
              file_alerts_src: Optional[pd.DataFrame] = None,
              rule_coverage_src: Optional[pd.DataFrame] = None):


    # Combined
    rc = rule_counts.reset_index()
    rc.columns = ["rule", "total_alerts"]
    rc.to_csv(out_dir / "rule_counts.csv", index=False)

    pr_df = project_rule.copy()
    pr_df.columns = ["project", "rule", "alerts"]
    pr_df.to_csv(out_dir / "project_rule_counts.csv", index=False)

    file_alerts.to_csv(out_dir / "file_alerts.csv", index=False)

    cov_df = (rule_coverage.reset_index().rename(columns={"project":"projects_affected"})
              if isinstance(rule_coverage, pd.Series) else rule_coverage)
    if cov_df is None or cov_df.empty:
        pd.DataFrame(columns=["rule","projects_affected"]).to_csv(out_dir / "rule_projects_affected.csv", index=False)
    else:
        cov_df.columns = ["rule","projects_affected"]
        cov_df.to_csv(out_dir / "rule_projects_affected.csv", index=False)

    # Optional per-source dumps
    if by_source:
        (rule_counts_src or pd.DataFrame(columns=["source","rule","total_alerts"])) \
            .to_csv(out_dir / "rule_counts_by_source.csv", index=False)
        (project_rule_src or pd.DataFrame(columns=["source","project","rule","alerts"])) \
            .to_csv(out_dir / "project_rule_counts_by_source.csv", index=False)
        (file_alerts_src or pd.DataFrame(columns=["source","project","file_path","rule","n_alerts"])) \
            .to_csv(out_dir / "file_alerts_by_source.csv", index=False)
        (rule_coverage_src or pd.DataFrame(columns=["source","rule","projects_affected"])) \
            .to_csv(out_dir / "rule_projects_affected_by_source.csv", index=False)



def write_summary_md(out_dir: Path, headline: dict, rule_counts: pd.Series, project_rule: pd.DataFrame):
    md = []
    md.append("# SpecDetect4AI — Aggregated Results\n")
    if headline:
        md.append(f"- **Projects analyzed**: {headline['projects']}")
        md.append(f"- **Files with alerts**: {headline['files_with_alerts']}")
        md.append(f"- **Total alerts**: {headline['total_alerts']}")
        md.append(f"- **Distinct rules**: {headline['distinct_rules']}\n")
    if not rule_counts.empty:
        md.append("## Most frequent rules\n")
        for r, c in rule_counts.head(10).items():
            md.append(f"- **{r}**: {int(c)}")
    if not project_rule.empty:
        proj_tot = project_rule.groupby("project")["n_alerts"].sum().sort_values(ascending=False).head(10)
        md.append("\n## Projects with the most alerts (Top 10)\n")
        for p, c in proj_tot.items():
            md.append(f"- **{p}**: {int(c)}")

        # NEW: Top rules by projects affected
        md.append("\n## Rules affecting the most projects (Top 10)\n")
        cov_top = compute_rule_coverage(project_rule).head(10)
        for r, k in cov_top.items():
            md.append(f"- **{r}**: {int(k)} projects")

    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _img_src(path: Path, inline: bool):
    """Return an <img> src attribute. If inline=True, embed as base64."""
    if not inline:
        return path.name  # relative link (assumes report next to images)
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return path.name


def write_html_report(out_dir: Path,
                      headline: dict,
                      rule_counts: pd.Series,
                      project_rule: pd.DataFrame,
                      inline_images: bool,
                      images: dict):
    """
    Generate report.html with simple styling + figures.
    images: dict name->Path  (e.g., {"top_rules": Path(...), ...})
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Tables (small preview)
    top_rules_tbl = rule_counts.head(20).reset_index()
    top_rules_tbl.columns = ["Rule", "Total Alerts"]

    proj_tot = project_rule.groupby("project")["n_alerts"].sum().sort_values(ascending=False).head(20).reset_index()
    proj_tot.columns = ["Project", "Total Alerts"]

    # NEW: coverage table (top 20 rules by projects affected)
    coverage_tbl = compute_rule_coverage(project_rule).head(20).reset_index()
    coverage_tbl.columns = ["Rule", "Projects Affected"]

    # HTML tables
    top_rules_html = top_rules_tbl.to_html(index=False, justify="left")
    proj_tot_html = proj_tot.to_html(index=False, justify="left")
    coverage_html = coverage_tbl.to_html(index=False, justify="left")

    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; line-height: 1.45; }
      h1 { font-size: 28px; margin-bottom: 0.2em; }
      h2 { font-size: 22px; margin-top: 1.4em; }
      .muted { color: #666; font-size: 14px; }
      .kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 8px; }
      .card { border: 1px solid #eee; border-radius: 10px; padding: 12px 14px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }
      img { width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; }
      th { background: #f7f7f7; text-align: left; }
      .small { font-size: 13px; }
      .footer { color: #888; margin-top: 28px; font-size: 12px; }
      .caption { color: #444; font-size: 14px; margin: 6px 0 14px; }
      .two { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
      @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
    </style>
    """
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>SpecDetect4AI — Report</title>{css}</head><body>"]
    html.append("<h1>SpecDetect4AI — Aggregated Report</h1>")
    html.append(f"<div class='muted'>Generated on {ts}</div>")

    # KPIs
    html.append("<div class='kpi'>")
    html.append(f"<div class='card'><b>Projects</b><br>{headline['projects']}</div>")
    html.append(f"<div class='card'><b>Files with alerts</b><br>{headline['files_with_alerts']}</div>")
    html.append(f"<div class='card'><b>Total alerts</b><br>{headline['total_alerts']}</div>")
    html.append(f"<div class='card'><b>Distinct rules</b><br>{headline['distinct_rules']}</div>")
    html.append("</div>")

    # Figures (grid)
    def fig_block(title, img_key, caption):
        p = images.get(img_key, None)
        if not p or not p.exists():
            return ""
        src = _img_src(p, inline_images)
        return f"<div><h2>{title}</h2><img src='{src}' alt='{img_key}'><div class='caption'>{caption}</div></div>"

    html.append("<div class='grid'>")
    html.append(fig_block("Top rules", "top_rules", "Most frequent rules by total alert count."))
    html.append(fig_block("Rules Pareto", "pareto_rules", "Frequency distribution with cumulative percentage."))
    html.append(fig_block("Top projects", "top_projects", "Projects with the highest alert counts (horizontal bars)."))
    html.append(fig_block("Heatmap: Rules × Projects", "heatmap_rules_projects", "Focused view on head of distributions."))
    html.append(fig_block("Rule coverage", "rules_coverage_scatter", "Per-rule frequency vs. number of affected projects."))
    html.append(fig_block("Projects affected by rule", "projects_affected", "Distinct projects impacted per rule (each project counted once per rule)."))
    html.append(fig_block("Per-rule dispersion", "per_rule_boxplot", "Distribution of alerts per project for dominant rules."))
    html.append("</div>")

    # Tables
    html.append("<h2>Summary tables</h2>")
    html.append("<div class='two'>")
    html.append(f"<div class='card'><h3>Top 20 rules</h3><div class='small'>{top_rules_html}</div></div>")
    html.append(f"<div class='card'><h3>Top 20 projects</h3><div class='small'>{proj_tot_html}</div></div>")
    html.append(f"<div class='card'><h3>Top 20 rules by projects affected</h3><div class='small'>{coverage_html}</div></div>")
    html.append("</div>")

    html.append("<div class='footer'>Report generated automatically from SpecDetect4AI JSON results.</div>")
    html.append("</body></html>")

    (out_dir / "report.html").write_text("".join(html), encoding="utf-8")


# ---------- Main ----------

from typing import List, Optional  # assure-toi que c'est bien importé en haut

def main():
    ap = argparse.ArgumentParser(description="Analyze SpecDetect4AI JSON outputs (CSV, charts, HTML).")
    ap.add_argument(
        "--input-dir", type=Path, nargs="+", required=True,
        help="One or more folders with JSON files (one per project)."
    )
    ap.add_argument(
        "--source-labels", type=str, nargs="+", default=None,
        help="Optional labels for each input dir (same order and length as --input-dir)."
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output folder for CSV/figures/report."
    )
    ap.add_argument("--top-n", type=int, default=15, help="Top-N rules for the main charts.")
    ap.add_argument("--heatmap-top-rules", type=int, default=8, help="#rules to display on the heatmap.")
    ap.add_argument("--heatmap-top-projects", type=int, default=12, help="#projects to display on the heatmap.")
    ap.add_argument("--top-projects", type=int, default=15, help="Top projects for the horizontal bar chart.")
    ap.add_argument("--inline-images", action="store_true",
                    help="Embed PNGs as base64 inside report.html (single self-contained file).")
    ap.add_argument("--by-source", dest="by_source", action="store_true",
                    help="Also compute per-source CSVs and comparison charts.")
    args = ap.parse_args()  # <-- PARSING AVANT UTILISATION

    # Charger les données (multi-dossiers), avec labels éventuels
    rows = load_reports(args.input_dir, args.source_labels)
    if not rows:
        print(f"[INFO] No valid JSON files found under: {', '.join(str(p) for p in args.input_dir)}")
        return

    # DataFrame plat avec colonne 'source'
    df = explode_to_rows(rows)

    # Agrégats combinés (on ignore 'source' pour les KPIs globaux)
    df_comb = df.drop(columns=["source"]) if "source" in df.columns else df.copy()
    rule_counts, project_rule_comb, file_alerts_comb, headline = compute_summaries(df_comb)

    # Couverture combinée (projets distincts par règle)
    rule_coverage = compute_rule_coverage(project_rule_comb)

    # Agrégats par source (si demandé)
    if args.by_source:
        rule_counts_src, project_rule_src, file_alerts_src = compute_summaries_by_source(df)
        rule_coverage_src = compute_rule_coverage_by_source(project_rule_src)
    else:
        rule_counts_src = project_rule_src = file_alerts_src = rule_coverage_src = None

    # Sorties tabulaires
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csvs(
        args.out_dir,
        rule_counts, project_rule_comb, file_alerts_comb, rule_coverage,
        by_source=args.by_source,
        rule_counts_src=rule_counts_src,
        project_rule_src=project_rule_src,
        file_alerts_src=file_alerts_src,
        rule_coverage_src=rule_coverage_src
    )

    # Figures (combinées)
    img_paths = {
        "top_rules": args.out_dir / "top_rules.png",
        "pareto_rules": args.out_dir / "pareto_rules.png",
        "top_projects": args.out_dir / "top_projects.png",
        "heatmap_rules_projects": args.out_dir / "heatmap_rules_projects.png",
        "rules_coverage_scatter": args.out_dir / "rules_coverage_scatter.png",
        "per_rule_boxplot": args.out_dir / "per_rule_boxplot.png",
        "projects_affected": args.out_dir / "projects_affected.png",
    }
    plot_top_rules(rule_counts, img_paths["top_rules"], top_n=args.top_n)
    plot_pareto(rule_counts, img_paths["pareto_rules"], top_n=max(args.top_n, 20))
    plot_top_projects(project_rule_comb, img_paths["top_projects"], top_projects=args.top_projects)
    plot_heatmap_rules_projects(
        project_rule_comb,
        img_paths["heatmap_rules_projects"],
        top_rules=args.heatmap_top_rules,
        top_projects=args.heatmap_top_projects,
    )
    plot_rules_coverage_scatter(project_rule_comb, img_paths["rules_coverage_scatter"], top_rules=max(20, args.top_n))
    plot_per_rule_boxplot(project_rule_comb, img_paths["per_rule_boxplot"], top_rules=max(8, min(15, args.top_n)))
    plot_projects_affected_bar(project_rule_comb, img_paths["projects_affected"], top_n=args.top_n)

    # Résumés & rapport HTML (combinés)
    write_summary_md(args.out_dir, headline, rule_counts, project_rule_comb)
    write_html_report(args.out_dir, headline, rule_counts, project_rule_comb, args.inline_images, img_paths)

    # Console
    print("=== Summary ===")
    print(f"Projects: {headline['projects']}, Files with alerts: {headline['files_with_alerts']}, "
          f"Total alerts: {headline['total_alerts']}, Distinct rules: {headline['distinct_rules']}")
    if not rule_counts.empty:
        print("\nTop rules:")
        for r, c in rule_counts.head(10).items():
            print(f"  {r}: {int(c)}")
    if not project_rule_comb.empty:
        cov_top = compute_rule_coverage(project_rule_comb).head(10)
        print("\nTop rules by projects affected:")
        for r, k in cov_top.items():
            print(f"  {r}: {int(k)} projects")
    if args.by_source and rule_counts_src is not None and not rule_counts_src.empty:
        print("\n[By-source CSVs saved: rule_counts_by_source.csv, project_rule_counts_by_source.csv, "
              "file_alerts_by_source.csv, rule_projects_affected_by_source.csv]")



if __name__ == "__main__":
    main()


# Example
#python extract_metrics.py \
#  --input-dir /data/sourceA/results_json /data/sourceB/results_json \
#  --source-labels SourceA SourceB \
#  --by-source \
#  --out-dir /tmp/extracted_metrics

