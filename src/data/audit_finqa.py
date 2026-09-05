import json
import re
from pathlib import Path
from collections import Counter


FINQA_DIR = Path("data/raw/finqa")


def load_json(filename):
    path = FINQA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_operation(program):
    if not program:
        return None
    return program.split("(", 1)[0]


def audit_split(filename):
    data = load_json(filename)

    print("\n" + "=" * 70)
    print(f"AUDIT: {filename}")
    print("=" * 70)
    print(f"Records: {len(data):,}")

    # ---------------------------------------------------------
    # 1. Missing / empty answers
    # ---------------------------------------------------------
    missing_answers = []
    empty_answers = []

    for record in data:
        qa = record.get("qa", {})
        answer = qa.get("answer")

        if answer is None:
            missing_answers.append(record["id"])
        elif isinstance(answer, str) and not answer.strip():
            empty_answers.append(record["id"])

    print("\n1. ANSWER COMPLETENESS")
    print(f"  Missing answer: {len(missing_answers):,}")
    print(f"  Empty answer:   {len(empty_answers):,}")

    # ---------------------------------------------------------
    # 2. Program completeness
    # ---------------------------------------------------------
    missing_programs = []

    for record in data:
        program = record.get("qa", {}).get("program")

        if not program or not program.strip():
            missing_programs.append(record["id"])

    print("\n2. PROGRAM COMPLETENESS")
    print(f"  Missing/empty programs: {len(missing_programs):,}")

    # ---------------------------------------------------------
    # 3. Operation distribution
    # ---------------------------------------------------------
    operations = Counter()

    for record in data:
        program = record.get("qa", {}).get("program")
        operation = get_operation(program)

        if operation:
            operations[operation] += 1

    print("\n3. OPERATIONS")

    for operation, count in operations.most_common():
        print(f"  {operation:25} {count:6,}")

    # ---------------------------------------------------------
    # 4. Reasoning step distribution
    # ---------------------------------------------------------
    steps = Counter()

    for record in data:
        reasoning_steps = record.get("qa", {}).get("steps", [])
        steps[len(reasoning_steps)] += 1

    print("\n4. REASONING STEPS")

    for step_count, count in sorted(steps.items()):
        print(f"  {step_count:2} steps: {count:6,}")

    # ---------------------------------------------------------
    # 5. Empty / missing questions
    # ---------------------------------------------------------
    bad_questions = []

    for record in data:
        question = record.get("qa", {}).get("question")

        if not question or not question.strip():
            bad_questions.append(record["id"])

    print("\n5. QUESTIONS")
    print(f"  Missing/empty questions: {len(bad_questions):,}")

    # ---------------------------------------------------------
    # 6. Duplicate questions
    # ---------------------------------------------------------
    question_to_ids = {}

    for record in data:
        question = record.get("qa", {}).get("question", "").strip().lower()

        question_to_ids.setdefault(question, []).append(record["id"])

    duplicate_groups = {
        q: ids
        for q, ids in question_to_ids.items()
        if q and len(ids) > 1
    }

    duplicate_records = sum(
        len(ids) for ids in duplicate_groups.values()
    )

    print("\n6. DUPLICATE QUESTIONS")
    print(f"  Duplicate question groups: {len(duplicate_groups):,}")
    print(f"  Records involved:           {duplicate_records:,}")

    if duplicate_groups:
        print("\n  Sample duplicate groups:")

        shown = 0

        for question, ids in duplicate_groups.items():
            print(f"    Question: {question}")
            print(f"    IDs:      {ids}")

            shown += 1

            if shown >= 5:
                break

    return data


def compare_splits(train, dev, test):
    print("\n" + "=" * 70)
    print("CROSS-SPLIT LEAKAGE AUDIT")
    print("=" * 70)

    split_data = {
        "train": train,
        "dev": dev,
        "test": test,
    }

    # ---------------------------------------------------------
    # IDs
    # ---------------------------------------------------------
    id_sets = {
        name: {record["id"] for record in records}
        for name, records in split_data.items()
    }

    print("\n1. ID OVERLAP")

    for first, second in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        overlap = id_sets[first] & id_sets[second]

        print(
            f"  {first} ∩ {second}: "
            f"{len(overlap):,}"
        )

    # ---------------------------------------------------------
    # Exact normalized question overlap
    # ---------------------------------------------------------
    question_sets = {}

    for name, records in split_data.items():
        question_sets[name] = {
            re.sub(r"\s+", " ", record["qa"]["question"].strip().lower())
            for record in records
            if record.get("qa", {}).get("question")
        }

    print("\n2. EXACT QUESTION OVERLAP")

    for first, second in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        overlap = question_sets[first] & question_sets[second]

        print(
            f"  {first} ∩ {second}: "
            f"{len(overlap):,}"
        )

        if overlap:
            print("    Sample:")
            for question in list(overlap)[:3]:
                print(f"      {question}")

    # ---------------------------------------------------------
    # Filename overlap
    # ---------------------------------------------------------
    filename_sets = {
        name: {
            record.get("filename")
            for record in records
            if record.get("filename")
        }
        for name, records in split_data.items()
    }

    print("\n3. SOURCE DOCUMENT OVERLAP")

    for first, second in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        overlap = filename_sets[first] & filename_sets[second]

        print(
            f"  {first} ∩ {second}: "
            f"{len(overlap):,} documents"
        )

        if overlap:
            print("    Sample:")
            for filename in list(overlap)[:3]:
                print(f"      {filename}")


def main():
    train = audit_split("train.json")
    dev = audit_split("dev.json")
    test = audit_split("test.json")

    compare_splits(train, dev, test)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()