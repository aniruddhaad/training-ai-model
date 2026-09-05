import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/ncert")

FILES = {
    "NCERT_11": DATA_DIR / "accounting_11.csv",
    "NCERT_12": DATA_DIR / "accounting_12.csv",
}


def profile(name, path):
    print("\n" + "=" * 80)
    print(f"{name}: {path}")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nMissing values:")
    missing = df.isna().sum()

    for col, count in missing.items():
        if count:
            print(f"  {col}: {count:,} ({count / len(df) * 100:.1f}%)")

    if missing.sum() == 0:
        print("  None")

    print("\nDuplicate rows:")
    print(f"  {df.duplicated().sum():,}")

    if "Question" in df.columns:
        print("\nDuplicate questions:")
        print(f"  {df['Question'].duplicated().sum():,}")
        print(f"  Unique questions: {df['Question'].nunique():,}")

    for column in [
        "Difficulty",
        "StudentLevel",
        "QuestionType",
        "QuestionComplexity",
        "grade",
        "subject",
    ]:
        if column in df.columns:
            print(f"\n{column} distribution:")
            counts = df[column].fillna("<MISSING>").value_counts()

            for value, count in counts.items():
                print(
                    f"  {str(value):35} "
                    f"{count:6,} "
                    f"({count / len(df) * 100:5.1f}%)"
                )

    if "Answer" in df.columns:
        answer_lengths = (
            df["Answer"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        print("\nAnswer length:")
        print(f"  Empty: {int((answer_lengths == 0).sum()):,}")
        print(f"  Mean: {answer_lengths.mean():.1f} chars")
        print(f"  Median: {answer_lengths.median():.1f} chars")
        print(f"  Max: {answer_lengths.max():,} chars")

    if "Question" in df.columns:
        q = df["Question"].fillna("").astype(str).str.lower()

        numerical_keywords = [
            "calculate",
            "calculation",
            "compute",
            "amount",
            "ratio",
            "percentage",
            "profit",
            "loss",
            "value",
            "journal",
            "balance",
            "depreciation",
            "interest",
            "capital",
            "goodwill",
            "shares",
            "partner",
        ]

        numerical_mask = q.apply(
            lambda x: any(keyword in x for keyword in numerical_keywords)
        )

        print("\nQuestion keyword profile:")
        print(
            f"  Questions containing numerical/accounting "
            f"keywords: {numerical_mask.sum():,} "
            f"({numerical_mask.mean() * 100:.1f}%)"
        )

    print("\nSample records:")
    sample_columns = [
        c for c in [
            "Topic",
            "Question",
            "Answer",
            "Difficulty",
            "QuestionType",
        ]
        if c in df.columns
    ]

    print(
        df[sample_columns]
        .head(3)
        .to_string(index=False)
    )


def main():
    print("=" * 80)
    print("FinMod — NCERT ACCOUNTING DATASET PROFILE")
    print("=" * 80)

    for name, path in FILES.items():
        if not path.exists():
            print(f"\nERROR: File not found: {path}")
            continue

        profile(name, path)

    print("\n" + "=" * 80)
    print("PROFILE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()