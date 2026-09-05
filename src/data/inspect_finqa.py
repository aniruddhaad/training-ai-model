import json
from pathlib import Path
from collections import Counter


FINQA_DIR = Path("data/raw/finqa")


def load_json(filename):
    path = FINQA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def profile_split(filename):
    data = load_json(filename)

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    print(f"Records: {len(data):,}")

    # ---------------------------------------------------------
    # Basic field presence
    # ---------------------------------------------------------
    required_fields = [
        "pre_text",
        "post_text",
        "filename",
        "table_ori",
        "table",
        "qa",
        "id",
    ]

    print("\nField presence:")
    for field in required_fields:
        count = sum(1 for r in data if field in r and r[field] is not None)
        print(f"  {field:20} {count:,}/{len(data):,}")

    # ---------------------------------------------------------
    # IDs and duplicate questions
    # ---------------------------------------------------------
    ids = [r.get("id") for r in data]
    questions = [
        r.get("qa", {}).get("question")
        for r in data
        if r.get("qa")
    ]

    print("\nUniqueness:")
    print(f"  Unique IDs:              {len(set(ids)):,}")
    print(f"  Duplicate IDs:           {len(ids) - len(set(ids)):,}")
    print(f"  Unique questions:        {len(set(questions)):,}")
    print(f"  Duplicate questions:     {len(questions) - len(set(questions)):,}")

    # ---------------------------------------------------------
    # Program / operation analysis
    # ---------------------------------------------------------
    programs = []
    operation_counts = Counter()
    step_counts = Counter()

    for record in data:
        qa = record.get("qa", {})

        program = qa.get("program")
        if program:
            programs.append(program)

            # Count the main operation.
            # Example:
            # subtract(5829, 5735)
            # -> subtract
            operation = program.split("(", 1)[0]
            operation_counts[operation] += 1

        steps = qa.get("steps", [])
        step_counts[len(steps)] += 1

    print("\nPrograms:")
    print(f"  Records with program:    {len(programs):,}")
    print(f"  Records without program: {len(data) - len(programs):,}")

    print("\nOperation frequency:")
    for operation, count in operation_counts.most_common():
        percentage = count / len(data) * 100
        print(f"  {operation:25} {count:6,} ({percentage:5.1f}%)")

    print("\nNumber of reasoning steps:")
    for steps, count in sorted(step_counts.items()):
        percentage = count / len(data) * 100
        print(f"  {steps:2} steps: {count:6,} ({percentage:5.1f}%)")

    # ---------------------------------------------------------
    # Answer / explanation analysis
    # ---------------------------------------------------------
    answer_types = Counter()
    explanation_present = 0

    for record in data:
        qa = record.get("qa", {})

        answer = qa.get("answer")

        if isinstance(answer, (int, float)):
            answer_types["numeric"] += 1
        elif isinstance(answer, str):
            answer_types["text"] += 1
        elif answer is None:
            answer_types["missing"] += 1
        else:
            answer_types[type(answer).__name__] += 1

        if qa.get("explanation"):
            explanation_present += 1

    print("\nAnswers:")
    for answer_type, count in answer_types.items():
        percentage = count / len(data) * 100
        print(f"  {answer_type:15} {count:6,} ({percentage:5.1f}%)")

    print(
        f"  With explanation:        "
        f"{explanation_present:,} "
        f"({explanation_present / len(data) * 100:.1f}%)"
    )

    # ---------------------------------------------------------
    # Table analysis
    # ---------------------------------------------------------
    table_rows = []
    table_cols = []

    for record in data:
        table = record.get("table", [])

        if table:
            table_rows.append(len(table))

            max_cols = max(
                (len(row) for row in table if isinstance(row, list)),
                default=0,
            )
            table_cols.append(max_cols)

    print("\nTables:")
    print(f"  Records with table:      {len(table_rows):,}")
    print(
        f"  Average rows:            "
        f"{sum(table_rows) / len(table_rows):.2f}"
        if table_rows
        else "  Average rows:            N/A"
    )
    print(
        f"  Average columns:         "
        f"{sum(table_cols) / len(table_cols):.2f}"
        if table_cols
        else "  Average columns:         N/A"
    )

    # ---------------------------------------------------------
    # Evidence analysis
    # ---------------------------------------------------------
    table_evidence = 0
    text_evidence = 0
    both_evidence = 0
    neither_evidence = 0

    for record in data:
        qa = record.get("qa", {})

        table_rows_used = qa.get("ann_table_rows", [])
        text_rows_used = qa.get("ann_text_rows", [])

        has_table = bool(table_rows_used)
        has_text = bool(text_rows_used)

        if has_table:
            table_evidence += 1

        if has_text:
            text_evidence += 1

        if has_table and has_text:
            both_evidence += 1

        if not has_table and not has_text:
            neither_evidence += 1

    print("\nEvidence usage:")
    print(
        f"  Table evidence only/used: {table_evidence:,} "
        f"({table_evidence / len(data) * 100:.1f}%)"
    )
    print(
        f"  Text evidence only/used:  {text_evidence:,} "
        f"({text_evidence / len(data) * 100:.1f}%)"
    )
    print(
        f"  Both table + text:        {both_evidence:,} "
        f"({both_evidence / len(data) * 100:.1f}%)"
    )
    print(
        f"  Neither:                  {neither_evidence:,} "
        f"({neither_evidence / len(data) * 100:.1f}%)"
    )

    # ---------------------------------------------------------
    # Sample programs
    # ---------------------------------------------------------
    print("\nSample reasoning programs:")

    shown = 0

    for record in data:
        qa = record.get("qa", {})
        program = qa.get("program")

        if program:
            print(f"\n  Question: {qa.get('question')}")
            print(f"  Program:  {program}")
            print(f"  Answer:   {qa.get('answer')}")

            shown += 1

            if shown >= 5:
                break


def main():
    print("=" * 70)
    print("FinQA DATASET PROFILE")
    print("=" * 70)

    for filename in ["train.json", "dev.json", "test.json"]:
        profile_split(filename)

    print("\n" + "=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()