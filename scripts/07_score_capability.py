import json
import re

EVAL_PATH = r"C:\FinMod\data\finance_eval_v1.jsonl"


def normalize(text):
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    return text


def contains_number(text, number):
    text = normalize(text)

    target = str(number)

    # Accept integers such as 175000 and 175000.0
    patterns = [
        rf"(?<!\d){re.escape(target)}(?!\d)",
        rf"(?<!\d){re.escape(target)}\.0+(?!\d)",
    ]

    return any(re.search(p, text) for p in patterns)


def expected_numbers(expected):
    return re.findall(r"\d+(?:\.\d+)?", normalize(expected))


with open(EVAL_PATH, "r", encoding="utf-8") as f:
    tests = [json.loads(line) for line in f if line.strip()]


print("=" * 80)
print("FINANCE EVALUATION DATASET")
print("=" * 80)

print(f"\nLoaded {len(tests)} evaluation examples.\n")

categories = {}

for test in tests:
    categories.setdefault(test["category"], 0)
    categories[test["category"]] += 1

for category, count in categories.items():
    print(f"{category}: {count}")

print("\n" + "=" * 80)
print("EXPECTED ANSWERS")
print("=" * 80)

for test in tests:
    print(f"\n[{test['id']}] {test['category']}")
    print(f"Q: {test['question']}")
    print(f"Expected: {test['expected_answer']}")

print("\n" + "=" * 80)
print("DATASET CHECK COMPLETE")
print("=" * 80)