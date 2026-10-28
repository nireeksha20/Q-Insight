
from pathlib import Path
import csv
import statistics

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "paired_evaluation.csv"
OUTPUT_CSV = ROOT / "data" / "ground_truth_evaluation.csv"
REPORT_TXT = ROOT / "data" / "ground_truth_summary.txt"


def main():
    if not INPUT_CSV.exists():
        print(f"Missing evaluation file: {INPUT_CSV}")
        print("Run python -u evaluate_pairs.py first.")
        return

    with INPUT_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    evaluated = []

    for row in rows:
        status = row.get("status", "")
        score_text = row.get("difference_score", "")
        changed_text = row.get("changed_pixel_fraction", "")

        usable = status == "aligned" and score_text not in ("", None)

        evaluated.append({
            "pair": row.get("image", ""),
            "clean_ground_truth": "good",
            "defective_ground_truth": "defective",
            "alignment_status": status,
            "difference_score": score_text if usable else "",
            "changed_pixel_fraction": changed_text if usable else "",
            "defect_region_review": "",
            "difference_highlights_actual_defect": "",
            "review_notes": (
                "Review heatmap against actual damage"
                if usable else "Alignment failed; inspect manually"
            ),
        })

    fields = [
        "pair",
        "clean_ground_truth",
        "defective_ground_truth",
        "alignment_status",
        "difference_score",
        "changed_pixel_fraction",
        "defect_region_review",
        "difference_highlights_actual_defect",
        "review_notes",
    ]

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evaluated)

    usable = [
        row for row in evaluated
        if row["alignment_status"] == "aligned"
        and row["difference_score"] != ""
    ]

    scores = [float(row["difference_score"]) for row in usable]
    changed = [
        float(row["changed_pixel_fraction"])
        for row in usable
        if row["changed_pixel_fraction"] != ""
    ]

    report = [
        "Q-INSIGHT GROUND-TRUTH REVIEW",
        "=" * 34,
        f"Matched pairs in dataset: {len(evaluated)}",
        f"Pairs with usable alignment: {len(usable)}",
        f"Pairs needing manual review: {len(evaluated) - len(usable)}",
        "",
        "KNOWN LABELS",
        "- Each good/defective folder pair is treated as a matched example.",
        "- The defective image is the positive example.",
        "- The clean image is its reference, not an independent negative test.",
        "",
        "VISUAL DIFFERENCE BASELINE",
    ]

    if scores:
        report.extend([
            f"Mean difference score: {statistics.mean(scores):.5f}",
            f"Median difference score: {statistics.median(scores):.5f}",
            f"Minimum difference score: {min(scores):.5f}",
            f"Maximum difference score: {max(scores):.5f}",
        ])

    if changed:
        report.append(
            f"Mean changed-pixel fraction: {statistics.mean(changed):.1%}"
        )

    report.extend([
        "",
        "METRICS NOT YET VALIDLY AVAILABLE",
        "- Accuracy, specificity, precision, recall and F1-score.",
        "- These require independent predictions and ground-truth outcomes",
        "  for both defective and non-defective inspection samples.",
        "",
        "NEXT DATA REQUIRED",
        "- Capture additional clean inspection images for each product type.",
        "- Include normal changes in lighting, angle and background.",
        "- Record whether each highlighted region overlaps the real defect.",
        "- Keep a separate test set that is not used to tune thresholds.",
        "",
        f"Review spreadsheet: {OUTPUT_CSV}",
    ])

    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nCSV: {OUTPUT_CSV}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
