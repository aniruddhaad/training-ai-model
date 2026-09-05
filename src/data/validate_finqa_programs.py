import json
import math
import re
from pathlib import Path


FINQA_DIR = Path("data/raw/finqa")


def load_json(filename):
    with open(FINQA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_number(value):
    """Convert a FinQA numeric value into a float."""
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    # Remove commas and dollar signs.
    value = value.replace(",", "").replace("$", "").strip()

    # FinQA sometimes represents negatives like:
    # -32 (32)
    # or simply -32
    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if not match:
        raise ValueError(f"Cannot parse number: {value}")

    return float(match.group())


def normalize_percentage(value):
    """Convert a percentage answer into its numeric percentage value."""
    value = str(value).strip()

    if "%" in value:
        return parse_number(value)

    return parse_number(value)


def close_enough(calculated, expected):
    """
    Compare two numeric values while allowing for rounding.
    """
    if math.isclose(calculated, expected, rel_tol=1e-3, abs_tol=1e-3):
        return True

    # FinQA answers are frequently rounded to one or two decimals.
    rounded_values = [
        round(calculated, 0),
        round(calculated, 1),
        round(calculated, 2),
    ]

    return any(
        math.isclose(v, expected, rel_tol=1e-3, abs_tol=1e-3)
        for v in rounded_values
    )


def resolve_arg(arg, variables):
    """
    Resolve a program argument.

    Examples:
        5829       -> 5829
        23.6%      -> 0.236
        const_100  -> 100
        #0         -> previous result
    """
    arg = arg.strip()

    if arg.startswith("#"):
        index = int(arg[1:])
        return variables[index]

    if arg.startswith("const_"):
        const_name = arg[6:]

        constants = {
            "100": 100.0,
            "1000": 1000.0,
            "1000000": 1000000.0,
        }

        if const_name not in constants:
            raise ValueError(f"Unknown constant: {arg}")

        return constants[const_name]

    if arg.endswith("%"):
        return parse_number(arg) / 100.0

    return parse_number(arg)


def execute_program(program):
    """
    Execute the arithmetic operations used by FinQA.

    Returns:
        calculated result
    """
    variables = []

    # Split program into individual operations.
    operations = [
        operation.strip()
        for operation in program.split("),")
    ]

    for operation in operations:
        operation = operation.strip()

        if not operation.endswith(")"):
            operation += ")"

        match = re.match(r"([a-zA-Z_]+)\((.*)\)", operation)

        if not match:
            raise ValueError(f"Cannot parse operation: {operation}")

        op = match.group(1)
        args_string = match.group(2)

        args = [
            arg.strip()
            for arg in args_string.split(",")
        ]

        values = [
            resolve_arg(arg, variables)
            for arg in args
        ]

        if op == "add":
            result = values[0] + values[1]

        elif op == "subtract":
            result = values[0] - values[1]

        elif op == "multiply":
            result = values[0] * values[1]

        elif op == "divide":
            if values[1] == 0:
                raise ZeroDivisionError()

            result = values[0] / values[1]

        elif op == "greater":
            result = values[0] > values[1]

        elif op == "table_sum":
            raise NotImplementedError("table_sum requires table context")

        elif op == "table_average":
            raise NotImplementedError("table_average requires table context")

        elif op == "table_max":
            raise NotImplementedError("table_max requires table context")

        elif op == "table_min":
            raise NotImplementedError("table_min requires table context")

        else:
            raise NotImplementedError(f"Unsupported operation: {op}")

        variables.append(result)

    return variables[-1]


def validate_record(record):
    qa = record["qa"]

    answer = qa.get("answer")
    program = qa.get("program")

    if not answer or not str(answer).strip():
        return "EMPTY_ANSWER", None, None

    if not program or not program.strip():
        return "EMPTY_PROGRAM", None, None

    try:
        calculated = execute_program(program)
    except NotImplementedError:
        return "UNSUPPORTED_OPERATION", None, None
    except Exception:
        return "PROGRAM_ERROR", None, None

    # Boolean result.
    if isinstance(calculated, bool):
        return "BOOLEAN", calculated, answer

    try:
        expected = normalize_percentage(answer)
    except Exception:
        return "ANSWER_PARSE_ERROR", calculated, answer

    if close_enough(calculated, expected):
        return "MATCH", calculated, expected

    # Some FinQA percentage answers represent
    # a decimal result as a percentage.
    if "%" in str(answer):
        percentage_result = calculated * 100

        if close_enough(percentage_result, expected):
            return "MATCH_PERCENT_CONVERSION", percentage_result, expected

    return "MISMATCH", calculated, expected


def validate_split(filename):
    data = load_json(filename)

    print("\n" + "=" * 70)
    print(f"VALIDATING: {filename}")
    print("=" * 70)

    counts = {
        "MATCH": 0,
        "MATCH_PERCENT_CONVERSION": 0,
        "BOOLEAN": 0,
        "EMPTY_ANSWER": 0,
        "EMPTY_PROGRAM": 0,
        "UNSUPPORTED_OPERATION": 0,
        "PROGRAM_ERROR": 0,
        "ANSWER_PARSE_ERROR": 0,
        "MISMATCH": 0,
    }

    mismatches = []

    for record in data:
        status, calculated, expected = validate_record(record)

        counts[status] += 1

        if status == "MISMATCH":
            mismatches.append(
                (
                    record["id"],
                    record["qa"]["question"],
                    record["qa"]["program"],
                    record["qa"]["answer"],
                    calculated,
                    expected,
                )
            )

    print("\nValidation results:")

    for status, count in counts.items():
        percentage = count / len(data) * 100
        print(
            f"  {status:28} "
            f"{count:6,} ({percentage:5.1f}%)"
        )

    if mismatches:
        print("\nSample mismatches:")

        for (
            record_id,
            question,
            program,
            answer,
            calculated,
            expected,
        ) in mismatches[:10]:

            print("\n  ID:", record_id)
            print("  Question:", question)
            print("  Program:", program)
            print("  FinQA answer:", answer)
            print("  Calculated:", calculated)
            print("  Expected numeric:", expected)

    return counts


def main():
    print("=" * 70)
    print("FinQA PROGRAM VALIDATION")
    print("=" * 70)

    totals = {}

    for filename in ["train.json", "dev.json", "test.json"]:
        totals[filename] = validate_split(filename)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()