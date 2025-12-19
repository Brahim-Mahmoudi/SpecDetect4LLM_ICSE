from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


INPUT_FILE = "Manual_Analysis.xlsx"
OUTPUT_CSV = "precision_by_smell.csv"


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _compute_precision_from_tp_fp(df: pd.DataFrame) -> dict:
    tp_col = _find_first_col(df, ["TP", "True Positive", "true_positive", "truepositives"])
    fp_col = _find_first_col(df, ["FP", "False Positive", "false_positive", "falsepositives"])

    if tp_col is None or fp_col is None:
        raise ValueError("TP and FP columns were not found")

    tp = pd.to_numeric(df[tp_col], errors="coerce")
    fp = pd.to_numeric(df[fp_col], errors="coerce")

    mask = tp.isin([0, 1]) & fp.isin([0, 1])
    tp_sum = int(tp[mask].sum())
    fp_sum = int(fp[mask].sum())

    denom = tp_sum + fp_sum
    precision = (tp_sum / denom) if denom > 0 else float("nan")

    return {"n_instances": int(mask.sum()), "TP": tp_sum, "FP": fp_sum, "precision": precision}


def _compute_precision_from_labels(df: pd.DataFrame) -> dict:
    label_col = _find_first_col(
        df, ["label", "labels", "result", "outcome", "judgement", "judgment", "status"]
    )
    if label_col is None:
        raise ValueError("No usable label column was found")

    labels = df[label_col].astype(str).str.strip().str.lower()
    tp_sum = int(labels.isin(["tp", "truepositive", "true positive", "1"]).sum())
    fp_sum = int(labels.isin(["fp", "falsepositive", "false positive", "0"]).sum())

    denom = tp_sum + fp_sum
    precision = (tp_sum / denom) if denom > 0 else float("nan")

    return {"n_instances": int(denom), "TP": tp_sum, "FP": fp_sum, "precision": precision}


def compute_precision(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    try:
        return _compute_precision_from_tp_fp(df)
    except Exception:
        return _compute_precision_from_labels(df)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    xlsx_path = Path(INPUT_FILE)
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"File not found: {xlsx_path}. Put {INPUT_FILE} in the same folder as this script."
        )

    xls = pd.ExcelFile(xlsx_path)
    rows: list[dict] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        stats = compute_precision(df)
        rows.append(
            {
                "smell": sheet.strip(),
                "n_instances": stats["n_instances"],
                "TP": stats["TP"],
                "FP": stats["FP"],
                "precision": stats["precision"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df["precision"] = out_df["precision"].map(safe_float)
    out_df = out_df.sort_values(by="smell").reset_index(drop=True)

    out_df.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)
    print(out_df.to_string(index=False))

    total_tp = int(out_df["TP"].sum())
    total_fp = int(out_df["FP"].sum())
    micro_denom = total_tp + total_fp
    micro_precision = (total_tp / micro_denom) if micro_denom > 0 else float("nan")

    valid_prec = out_df["precision"].dropna()
    macro_precision = float(valid_prec.mean()) if len(valid_prec) else float("nan")

    best_row = None
    worst_row = None
    if len(valid_prec):
        best_idx = out_df["precision"].idxmax()
        worst_idx = out_df["precision"].idxmin()
        best_row = out_df.loc[best_idx]
        worst_row = out_df.loc[worst_idx]

    print("\nSummary")
    print(f"Number of smells: {len(out_df)}")
    print(f"Total usable instances (TP + FP): {micro_denom}")

    if not math.isnan(micro_precision):
        print(f"Micro average precision: {micro_precision:.4f}")
    else:
        print("Micro average precision: NaN")

    if not math.isnan(macro_precision):
        print(f"Macro average precision: {macro_precision:.4f}")
    else:
        print("Macro average precision: NaN")

    if best_row is not None and worst_row is not None:
        bp = float(best_row["precision"])
        wp = float(worst_row["precision"])
        bp_str = f"{bp:.4f}" if not math.isnan(bp) else "NaN"
        wp_str = f"{wp:.4f}" if not math.isnan(wp) else "NaN"

        print(
            f"Best precision: {best_row['smell']} with {bp_str} using TP={int(best_row['TP'])} FP={int(best_row['FP'])}"
        )
        print(
            f"Worst precision: {worst_row['smell']} with {wp_str} using TP={int(worst_row['TP'])} FP={int(worst_row['FP'])}"
        )

    print(f"\nCSV written: {Path(OUTPUT_CSV).resolve()}")


if __name__ == "__main__":
    main()

