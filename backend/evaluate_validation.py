
from pathlib import Path
import csv
import statistics

from evaluate_pairs import (
    GOOD_DIR,
    BAD_DIR,
    OUT_DIR,
    CSV_PATH,
    read_image,
    shared_center_crop,
    resize_pair,
    align_pair,
    compare_pair,
)

ROOT = Path(__file__).resolve().parent.parent
REPORT_CSV = ROOT / "data" / "validation_metrics.csv"
SUMMARY_TXT = ROOT / "data" / "validation_summary.txt"


def main():
    results = []

    for i in range(1, 17):
        good_path = GOOD_DIR / f"image{i}.jpg"
        bad_path = BAD_DIR / f"image{i}.jpg"

        if not good_path.exists() or not bad_path.exists():
            print(f"Pair {i}: missing image; skipped")
            continue

        try:
            good = read_image(good_path)
            bad = read_image(bad_path)

            good, bad = shared_center_crop(good, bad)
            good, bad = resize_pair(good, bad)

            aligned_bad, valid, method, status = align_pair(good, bad)

            result = {
                "pair": i,
                "alignment": method,
                "status": status,
                "difference_score": "",
                "changed_pixel_fraction": "",
            }

            if status == "aligned":
                _, _, score, changed = compare_pair(
                    good, aligned_bad, valid
                )
                result["difference_score"] = round(score, 5)
                result["changed_pixel_fraction"] = round(changed, 5)

                print(
                    f"Pair {i:02}: score={score:.5f}, "
                    f"changed={changed:.1%}, alignment={method}"
                )
            else:
                print(f"Pair {i:02}: manual review required")

            results.append(result)

        except Exception as exc:
            print(f"Pair {i:02}: ERROR - {exc}")
            results.append({
                "pair": i,
                "alignment": "error",
                "status": "error",
                "difference_score": "",
                "changed_pixel_fraction": "",
            })

    fields = [
        "pair", "alignment", "status",
        "difference_score", "changed_pixel_fraction",
    ]

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    usable = [
        r for r in results
        if r["status"] == "aligned"
        and r["difference_score"] != ""
    ]

    scores = [float(r["difference_score"]) for r in usable]
    changed_values = [
        float(r["changed_pixel_fraction"]) for r in usable
    ]

    ranked = sorted(
        usable,
        key=lambda r: float(r["difference_score"]),
        reverse=True,
    )

    lines = [
        "Q-INSIGHT VALIDATION SUMMARY",
        "=" * 32,
        f"Pairs available: {len(results)}",
        f"Successfully aligned: {len(usable)}",
        f"Manual review/errors: {len(results) - len(usable)}",
        "",
        "VISUAL DIFFERENCE METRICS",
        "-" * 24,
    ]

    if scores:
        lines.extend([
            f"Mean difference score: {statistics.mean(scores):.5f}",
            f"Median difference score: {statistics.median(scores):.5f}",
            f"Minimum difference score: {min(scores):.5f}",
            f"Maximum difference score: {max(scores):.5f}",
            f"Mean changed-pixel fraction: {statistics.mean(changed_values):.1%}",
            "",
            "PAIRS RANKED BY DIFFERENCE SCORE",
            "(Higher scores indicate more visual difference, not necessarily more damage)",
        ])

        for rank, row in enumerate(ranked, start=1):
            lines.append(
                f"{rank:02}. Pair {row['pair']:02}: "
                f"score={float(row['difference_score']):.5f}, "
                f"changed={float(row['changed_pixel_fraction']):.1%}"
            )
    else:
        lines.append("No pairs aligned successfully; no aggregate metrics available.")

    lines.extend([
        "",
        "LIMITATIONS",
        "- These are visual-difference metrics, not calibrated defect probabilities.",
        "- A high score can result from lighting, perspective, or texture differences.",
        "- Alignment failures are excluded from aggregate score statistics.",
        "- Detection accuracy, precision, recall, and F1 require ground-truth defect labels",
        "  and a defined prediction rule evaluated on suitable validation data.",
    ])

    summary = "\n".join(lines)
    SUMMARY_TXT.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"\nCSV report: {REPORT_CSV}")
    print(f"Summary report: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
