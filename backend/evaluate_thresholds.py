
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "paired_evaluation.csv"
OUTPUT_CSV = ROOT / "data" / "threshold_evaluation.csv"
REPORT_TXT = ROOT / "data" / "threshold_summary.txt"

# Candidate thresholds for exploration, not production settings.
THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]


def main():
    if not INPUT_CSV.exists():
        print(f"Missing input: {INPUT_CSV}")
        print("Run python -u evaluate_pairs.py first.")
        return

    with INPUT_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    usable = []
    manual_review = 0

    for row in rows:
        if (
            row.get("status") == "aligned"
            and row.get("difference_score", "") not in ("", None)
        ):
            usable.append({
                "image": row.get("image", ""),
                "score": float(row["difference_score"]),
                "changed": row.get("changed_pixel_fraction", ""),
            })
        else:
            manual_review += 1

    if not usable:
        print("No aligned pairs with valid scores.")
        return

    output_rows = []
    report = [
        "Q-INSIGHT CANDIDATE THRESHOLD EVALUATION",
        "=" * 42,
        f"Total matched pairs: {len(rows)}",
        f"Aligned pairs: {len(usable)}",
        f"Manual review required: {manual_review}",
        "",
        "A pair is flagged when its visual-difference score is >= threshold.",
        "These scores compare a defective image with its clean reference.",
        "They are NOT independent per-image defect predictions.",
        "",
        "THRESHOLD RESULTS",
        "-" * 18,
        "Threshold | Flagged | Not flagged | Coverage of aligned pairs",
    ]

    for threshold in THRESHOLDS:
        flagged = [
            item for item in usable
            if item["score"] >= threshold
        ]
        not_flagged = [
            item for item in usable
            if item["score"] < threshold
        ]

        coverage = len(flagged) / len(usable)

        output_rows.append({
            "threshold": threshold,
            "aligned_pairs": len(usable),
            "flagged_pairs": len(flagged),
            "not_flagged_pairs": len(not_flagged),
            "coverage_fraction": round(coverage, 4),
            "flagged_pair_ids": "; ".join(
                item["image"] for item in flagged
            ),
            "not_flagged_pair_ids": "; ".join(
                item["image"] for item in not_flagged
            ),
            "manual_review_pairs": manual_review,
        })

        report.append(
            f"{threshold:8.3f} | {len(flagged):7} | "
            f"{len(not_flagged):11} | {coverage:.1%}"
        )

    report.extend([
        "",
        "INTERPRETATION",
        "- Higher thresholds flag fewer paired comparisons.",
        "- Lower thresholds flag more paired comparisons.",
        "- Coverage is not accuracy, recall, precision, or F1-score.",
        "- These are all known defective-versus-clean comparisons.",
        "- No independent clean inspection comparisons are included.",
        "- Alignment failures remain manual-review cases.",
        "",
        "NEXT REQUIREMENT",
        "Collect independent clean inspection images under normal variation,",
        "then evaluate complete inspections using a fixed threshold.",
        "Choose the final threshold on validation data and assess it on a",
        "separate test set that was not used to tune the system.",
    ])

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(output_rows)

    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nCSV: {OUTPUT_CSV}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
