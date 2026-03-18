from pathlib import Path
import csv


def parse_mean_sd(text: str):
    s = text.replace("*", "").strip()
    if "±" in s:
        left, right = s.split("±", 1)
        return float(left.strip()), float(right.strip())
    return float(s), 0.0


def read_md_table(md_path: Path):
    lines = [ln.strip() for ln in md_path.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        raise ValueError("Markdown table looks empty or invalid.")

    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    data_lines = lines[2:]  # skip alignment row

    rows = []
    for ln in data_lines:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) != len(headers):
            continue
        rows.append(dict(zip(headers, parts)))
    return rows


def model_name(rep: str, backbone: str):
    rep_fix = rep.replace("Raw (1D)", "Raw(1D)")
    return f"{rep_fix}-{backbone}"


def main():
    base_dir = Path(__file__).resolve().parent
    md_path = base_dir / "baseline_performance_table.md"
    out_csv = base_dir / "baseline_performance_top5.csv"

    rows = read_md_table(md_path)

    parsed = []
    for r in rows:
        parsed.append(
            {
                "input_family": r["Input family"],
                "representation": r["Representation"],
                "backbone": r["Backbone"].replace("*", "").strip(),
                "model": model_name(r["Representation"], r["Backbone"].replace("*", "").strip()),
                "binary-f1": parse_mean_sd(r["Binary F1"]),
                "binary-recall": parse_mean_sd(r["Binary Recall"]),
                "macro-f1": parse_mean_sd(r["4-class Macro-F1"]),
                "macro-recall": parse_mean_sd(r["4-class Macro-Recall"]),
            }
        )

    out_rows = []
    for metric_key in ["binary-f1", "binary-recall", "macro-f1", "macro-recall"]:
        ranked = sorted(parsed, key=lambda x: x[metric_key][0], reverse=True)[:5]
        for i, item in enumerate(ranked, start=1):
            mean, sd = item[metric_key]
            out_rows.append(
                {
                    "metric": metric_key,
                    "rank": i,
                    "model": item["model"],
                    "input_family": item["input_family"],
                    "representation": item["representation"],
                    "backbone": item["backbone"],
                    "mean": f"{mean:.4f}",
                    "sd": f"{sd:.4f}",
                    "mean_sd": f"{mean:.4f} ± {sd:.4f}",
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "rank",
                "model",
                "input_family",
                "representation",
                "backbone",
                "mean",
                "sd",
                "mean_sd",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
