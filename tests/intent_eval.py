"""
Lightweight intent & slot evaluation harness for the TLC chatbot.

Extend the TEST_CASES list below with new examples to track regressions.
Run via:
    python tests/intent_eval.py --show-misclassified
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from pprint import pprint
import importlib.util

# Skip pip bootstrapping inside app.py during evaluations.
os.environ.setdefault("TLC_SKIP_PIP", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Add more coverage here as intents/slots evolve.
TEST_CASES = [
    {
        "text": "Lihat katalog training leadership dan safety",
        "expected_intent": "catalog",
        "expected_slots": {},
    },
    {
        "text": "Jadwal JKK-101 batch berikutnya kapan?",
        "expected_intent": "schedule",
        "expected_slots": {"course": "JKK-101"},
    },
    {
        "text": "Berapa biaya training JKK-SV-101 untuk 10 orang?",
        "expected_intent": "pricing",
        "expected_slots": {"course": "JKK-SV-101", "pax": 10},
    },
    {
        "text": "Saya mau daftar kelas TCLASS-201 bulan depan",
        "expected_intent": "registration",
        "expected_slots": {"course": "TCLASS-201"},
    },
    {
        "text": "Kalau saya mau request in-house JKK untuk 25 orang gimana prosesnya?",
        "expected_intent": "external_training_request",
        "expected_slots": {"course": "JKK", "pax": 25},
    },
    {
        "text": "Bisa diadakan di luar pabrik nggak? 20 peserta di Karawang",
        "expected_intent": "external_training_request",
        "expected_slots": {"pax": 20, "location": "Karawang"},
    },
    {
        "text": "Kebijakan pembatalan kalau mau refund bagaimana?",
        "expected_intent": "policy",
        "expected_slots": {},
    },
    {
        "text": "Apakah topik bisa dikustom sesuai kebutuhan perusahaan?",
        "expected_intent": "custom",
        "expected_slots": {},
    },
]


REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "sklearn",
    "sentence_transformers",
    "gradio",
]


def evaluate_cases(cases, detect_intent_fn, extract_slots_fn, show_misclassified=False):
    total = len(cases)
    intent_correct = 0
    per_intent = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(Counter)
    slot_match_count = 0
    slot_expectation_count = 0
    misclassified = []

    for case in cases:
        text = case["text"]
        expected_intent = case["expected_intent"]
        expected_slots = case.get("expected_slots", {})

        predicted_intent, debug = detect_intent_fn(text)
        predicted_intent = predicted_intent or "other"
        predicted_slots = extract_slots_fn(text)

        per_intent[expected_intent]["total"] += 1
        confusion[expected_intent][predicted_intent] += 1

        if predicted_intent == expected_intent:
            intent_correct += 1
            per_intent[expected_intent]["correct"] += 1
        else:
            misclassified.append(
                {
                    "text": text,
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "debug": debug,
                }
            )

        for key, expected_value in expected_slots.items():
            slot_expectation_count += 1
            if predicted_slots.get(key) == expected_value:
                slot_match_count += 1

    overall_intent_acc = intent_correct / total if total else 0.0
    slot_acc = (
        slot_match_count / slot_expectation_count if slot_expectation_count else 1.0
    )

    print(f"Total cases: {total}")
    print(f"Overall intent accuracy: {overall_intent_acc:.2%}")
    print(f"Slot match rate (only expected slots are scored): {slot_acc:.2%}")
    print("\nPer-intent accuracy:")
    for intent_name, stats in sorted(per_intent.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"  - {intent_name}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    print("\nConfusion matrix (rows=expected, cols=predicted):")
    for expected_intent, preds in sorted(confusion.items()):
        row_total = sum(preds.values())
        row_parts = ", ".join(
            f"{pred}:{count}" for pred, count in preds.most_common()
        )
        print(f"  {expected_intent} [{row_total}]: {row_parts}")

    if show_misclassified and misclassified:
        print("\nMisclassified examples:")
        pprint(misclassified)
    elif show_misclassified:
        print("\nNo misclassifications in this run.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TLC chatbot intents & slots")
    parser.add_argument(
        "--show-misclassified",
        action="store_true",
        help="Print misclassified examples with debug info",
    )
    args = parser.parse_args()

    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        print("⚠️  Cannot run evaluation. Missing dependencies:", ", ".join(missing))
        print("Install requirements.txt or run the full app environment, then re-run this script.")
        return

    from app import detect_intent, extract_slots

    evaluate_cases(
        TEST_CASES,
        detect_intent_fn=detect_intent,
        extract_slots_fn=extract_slots,
        show_misclassified=args.show_misclassified,
    )


if __name__ == "__main__":
    main()
