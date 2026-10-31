
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "patchcore_audit_v2_predictions.csv"

PRIMARY_PAIRS = {1, 2, 3, 11}
SEPARATE_PAIRS = {16}
EXCLUDED_PAIRS = {10}
THRESHOLD = 0.5


def load_predictions():
    records = []

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            normalized_path = row["path"].replace("\\", "/").lower()
            parts = normalized_path.split("/")

            if "defective" in parts:
                label = 1
            elif "good" in parts:
                label = 0
            else:
                print(f"Skipping unrecognized class: {row['path']}")
                continue

            filename = row["image"].lower()
            pair_number = int(
                filename.replace("image", "").replace(".jpg", "")
            )

            records.append({
                "pair": pair_number,
                "image": row["image"],
                "class": "DEFECTIVE" if label else "GOOD",
                "actual": label,
                "score": float(row["score"]),
            })

    return records


def calculate_auc(records):
    positives = [r["score"] for r in records if r["actual"] == 1]
    negatives = [r["score"] for r in records if r["actual"] == 0]

    if not positives or not negatives:
        return None

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1
            elif positive == negative:
                wins += 0.5

    return wins / (len(positives) * len(negatives))


def evaluate(records, title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    if not records:
        print("No predictions found.")
        return

    print(f"Images: {len(records)}")
    print("\nIndividual predictions:")

    for r in sorted(records, key=lambda x: (x["pair"], x["actual"])):
        predicted = "DEFECTIVE" if r["score"] >= THRESHOLD else "GOOD"
        result = "CORRECT" if predicted == r["class"] else "WRONG"

        print(
            f"Pair {r['pair']:>2} | {r['class']:<9} | "
            f"score={r['score']:.6f} | "
            f"predicted={predicted:<9} | {result}"
        )

    tp = sum(
        r["actual"] == 1 and r["score"] >= THRESHOLD for r in records
    )
    fp = sum(
        r["actual"] == 0 and r["score"] >= THRESHOLD for r in records
    )
    tn = sum(
        r["actual"] == 0 and r["score"] < THRESHOLD for r in records
    )
    fn = sum(
        r["actual"] == 1 and r["score"] < THRESHOLD for r in records
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    auc = calculate_auc(records)

    print(f"\nThreshold: {THRESHOLD}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUROC:     {auc:.4f}" if auc is not None else "AUROC: N/A")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {CSV_PATH}")

    records = load_predictions()

    primary = [r for r in records if r["pair"] in PRIMARY_PAIRS]
    separate = [r for r in records if r["pair"] in SEPARATE_PAIRS]
    excluded = [r for r in records if r["pair"] in EXCLUDED_PAIRS]

    evaluate(primary, "PRIMARY EVALUATION: PAIRS 1, 2, 3, 11")
    evaluate(separate, "SEPARATE CHECK: PAIR 16")

    print(f"\nPair 10 excluded from primary evaluation: {len(excluded)} rows")
    print("No model training or source-image changes were performed.")


if __name__ == "__main__":
    main()
