import json
import os
import re


# ============================================================
# Paths
# ============================================================

BASE_RESULTS = (
    r"C:\FinMod\outputs\finance_eval_v1_base_results.jsonl"
)

LORA_RESULTS = (
    r"C:\FinMod\outputs\finance_eval_v1_r16_qkvo_v4_results.jsonl"
)


# ============================================================
# Benchmark scoring
#
# This is deliberately NOT an LLM judge.
#
# We use explicit rules for the numerical/accounting answers
# and manual semantic scoring rules for conceptual answers.
# ============================================================


def load_jsonl(path):
    results = {}

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            results[item["id"]] = item

    return results


def normalize(text):
    """Normalize text for simple keyword checks."""

    text = text.lower()

    # Normalize common encoding artifacts from the terminal.
    text = text.replace("╫", "x")
    text = text.replace("æ", "a")

    # Remove punctuation.
    text = re.sub(r"[^a-z0-9$%+\-./ ]+", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_any(text, terms):
    return any(term in text for term in terms)


# ============================================================
# Individual scoring rules
# ============================================================

def score_answer(example_id, answer):

    text = normalize(answer)

    # --------------------------------------------------------
    # A1 Revenue
    # --------------------------------------------------------

    if example_id == "A1":

        return (
            contains_any(text, [
                "income",
                "revenue",
                "money",
                "amount of money"
            ])
            and contains_any(text, [
                "selling",
                "sale",
                "products",
                "services",
                "goods"
            ])
        )

    # --------------------------------------------------------
    # A2 Asset
    # --------------------------------------------------------

    if example_id == "A2":

        return (
            contains_any(text, [
                "resource"
            ])
            and contains_any(text, [
                "controlled",
                "owned"
            ])
            and contains_any(text, [
                "future economic benefits",
                "future benefit",
                "economic benefit"
            ])
        )

    # --------------------------------------------------------
    # A3 Liability
    # --------------------------------------------------------

    if example_id == "A3":

        return (
            contains_any(text, [
                "obligation",
                "liability"
            ])
            and contains_any(text, [
                "pay",
                "outflow",
                "economic resources"
            ])
        )

    # --------------------------------------------------------
    # A4 Gross vs Net Profit
    # --------------------------------------------------------

    if example_id == "A4":

        gross_ok = (
            contains_any(text, ["gross profit"])
            and contains_any(text, ["revenue"])
            and contains_any(text, [
                "cost of goods sold",
                "cogs"
            ])
        )

        net_ok = (
            contains_any(text, ["net profit"])
            and contains_any(text, [
                "operating expenses",
                "interest",
                "taxes",
                "tax"
            ])
        )

        return gross_ok and net_ok

    # --------------------------------------------------------
    # A5 Positive profit but negative OCF
    # --------------------------------------------------------

    if example_id == "A5":

        ar_ok = contains_any(text, [
            "accounts receivable",
            "receivable",
            "cash not collected",
            "not collected"
        ])

        inventory_ok = contains_any(text, [
            "inventory",
            "working capital"
        ])

        negative_cash_ok = contains_any(text, [
            "negative operating cash flow",
            "negative cash flow",
            "reduces cash",
            "consume cash"
        ])

        return (
            negative_cash_ok
            and (ar_ok or inventory_ok)
        )

    # --------------------------------------------------------
    # B1 Working Capital formula
    # --------------------------------------------------------

    if example_id == "B1":

        return (
            "current assets" in text
            and "current liabilities" in text
            and (
                "-" in text
                or "difference" in text
                or "subtract" in text
            )
        )

    # --------------------------------------------------------
    # B2 Working Capital = 175,000
    # --------------------------------------------------------

    if example_id == "B2":

        return contains_any(text, [
            "175,000",
            "175000",
            "175k"
        ])

    # --------------------------------------------------------
    # B3 Gross Margin = 25%
    # --------------------------------------------------------

    if example_id == "B3":

        return (
            contains_any(text, [
                "25%",
                "25 percent"
            ])
            and contains_any(text, [
                "gross margin"
            ])
        )

    # --------------------------------------------------------
    # B4 Percentage increase = 25%
    # --------------------------------------------------------

    if example_id == "B4":

        return contains_any(text, [
            "25%",
            "25 percent"
        ])

    # --------------------------------------------------------
    # B5 Percentage decrease = 25%
    # --------------------------------------------------------

    if example_id == "B5":

        return contains_any(text, [
            "25%",
            "25 percent"
        ])

    # --------------------------------------------------------
    # C1 Bank loan
    #
    # Assets +120K
    # Liabilities +120K
    # Equity unchanged
    # --------------------------------------------------------

    if example_id == "C1":

        assets_ok = contains_any(text, [
            "assets increase",
            "assets are increased",
            "assets +120",
            "assets + $120"
        ])

        liabilities_ok = contains_any(text, [
            "liabilities increase",
            "liabilities are increased",
            "liabilities +120",
            "liabilities + $120"
        ])

        equity_ok = contains_any(text, [
            "equity does not change",
            "equity unchanged",
            "equity remains unchanged"
        ])

        return assets_ok and liabilities_ok and equity_ok

    # --------------------------------------------------------
    # C2 Owner investment
    #
    # Assets +80K
    # Equity +80K
    # Liabilities unchanged
    # --------------------------------------------------------

    if example_id == "C2":

        assets_ok = contains_any(text, [
            "assets increase",
            "assets are increased",
            "assets = $80,000"
        ])

        equity_ok = contains_any(text, [
            "equity increase",
            "equity increases",
            "equity = $80,000"
        ])

        liabilities_ok = (
            "liabilities do not change" in text
            or "liabilities unchanged" in text
            or "liabilities = $0" in text
        )

        return assets_ok and equity_ok and liabilities_ok

    # --------------------------------------------------------
    # C3 Loan repayment
    # --------------------------------------------------------

    if example_id == "C3":

        assets_ok = contains_any(text, [
            "assets decrease",
            "assets are decreased",
            "assets -25"
        ])

        liabilities_ok = contains_any(text, [
            "liabilities decrease",
            "liabilities are decreased",
            "liabilities -25"
        ])

        equity_ok = contains_any(text, [
            "equity does not change",
            "equity unchanged",
            "equity remains unchanged"
        ])

        return assets_ok and liabilities_ok and equity_ok

    # --------------------------------------------------------
    # C4 Equipment for cash
    #
    # Cash -60K
    # Equipment +60K
    # Total assets unchanged
    # Liabilities unchanged
    # Equity unchanged
    # --------------------------------------------------------

    if example_id == "C4":

        cash_ok = contains_any(text, [
            "cash decreases",
            "cash decreased",
            "cash -60",
            "cash decreases by $60"
        ])

        equipment_ok = contains_any(text, [
            "equipment increases",
            "equipment increased",
            "equipment +60"
        ])

        total_assets_ok = contains_any(text, [
            "total assets do not change",
            "total assets unchanged",
            "total assets remain unchanged"
        ])

        liabilities_ok = contains_any(text, [
            "liabilities do not change",
            "liabilities unchanged"
        ])

        equity_ok = contains_any(text, [
            "equity does not change",
            "equity unchanged"
        ])

        return (
            cash_ok
            and equipment_ok
            and total_assets_ok
            and liabilities_ok
            and equity_ok
        )

    # --------------------------------------------------------
    # C5 Cash revenue
    #
    # Assets +70K
    # Liabilities unchanged
    # Equity +70K
    # --------------------------------------------------------

    if example_id == "C5":

        assets_ok = contains_any(text, [
            "assets increase",
            "assets are increased",
            "assets +70"
        ])

        liabilities_ok = contains_any(text, [
            "liabilities do not change",
            "liabilities unchanged"
        ])

        equity_ok = contains_any(text, [
            "equity increases",
            "equity increased",
            "equity +70"
        ])

        return assets_ok and liabilities_ok and equity_ok

    # --------------------------------------------------------
    # Arithmetic
    # --------------------------------------------------------

    if example_id == "D1":
        return contains_any(text, ["83"])

    if example_id == "D2":
        return contains_any(text, ["82"])

    if example_id == "D3":
        return contains_any(text, ["378"])

    if example_id == "D4":
        return contains_any(text, ["14"])

    if example_id == "D5":
        return contains_any(text, ["28"])

    # --------------------------------------------------------
    # E1
    #
    # GP = 600K
    # OP = 250K
    # --------------------------------------------------------

    if example_id == "E1":

        return (
            contains_any(text, [
                "$600,000",
                "600,000",
                "600000",
                "600k"
            ])
            and
            contains_any(text, [
                "$250,000",
                "250,000",
                "250000",
                "250k"
            ])
        )

    # --------------------------------------------------------
    # E2
    #
    # New WC = 200K
    # --------------------------------------------------------

    if example_id == "E2":

        return contains_any(text, [
            "200,000",
            "200000",
            "200k"
        ])

    # --------------------------------------------------------
    # E3
    #
    # OCF = 320K
    # --------------------------------------------------------

    if example_id == "E3":

        return contains_any(text, [
            "320,000",
            "320000",
            "320k"
        ])

    # --------------------------------------------------------
    # E4
    #
    # OCF reduced by 150K
    # --------------------------------------------------------

    if example_id == "E4":

        reduction_ok = contains_any(text, [
            "reduces operating cash flow",
            "reduce operating cash flow",
            "reduces cash flow",
            "negative",
            "decrease"
        ])

        amount_ok = contains_any(text, [
            "150,000",
            "150000",
            "150k"
        ])

        return reduction_ok and amount_ok

    # --------------------------------------------------------
    # E5
    #
    # GP increases by 300K
    # --------------------------------------------------------

    if example_id == "E5":

        increase_ok = contains_any(text, [
            "increases",
            "increase",
            "increased"
        ])

        amount_ok = contains_any(text, [
            "300,000",
            "300000",
            "300k",
            "0.3 million"
        ])

        return increase_ok and amount_ok

    raise ValueError(
        f"No scoring rule exists for {example_id}"
    )


# ============================================================
# Score one result set
# ============================================================

def score_results(results):

    scored = []

    for example_id, item in results.items():

        passed = score_answer(
            example_id,
            item["model_answer"]
        )

        scored.append({
            "id": example_id,
            "category": item["category"],
            "question": item["question"],
            "model_answer": item["model_answer"],
            "correct": passed
        })

    return scored


# ============================================================
# Summary
# ============================================================

def summarize(scored):

    categories = {}

    for item in scored:

        category = item["category"]

        if category not in categories:

            categories[category] = {
                "correct": 0,
                "total": 0
            }

        categories[category]["total"] += 1

        if item["correct"]:
            categories[category]["correct"] += 1

    total_correct = sum(
        x["correct"]
        for x in categories.values()
    )

    total_questions = sum(
        x["total"]
        for x in categories.values()
    )

    return categories, total_correct, total_questions


# ============================================================
# Print detailed comparison
# ============================================================

def main():

    print("=" * 70)
    print("Finance Evaluation v1 - BASE vs LoRA")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    base_results = load_jsonl(BASE_RESULTS)

    lora_results = load_jsonl(LORA_RESULTS)

    print(f"Base results loaded: {len(base_results)}")
    print(f"LoRA results loaded:  {len(lora_results)}")
    print()

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    base_scored = score_results(base_results)

    lora_scored = score_results(lora_results)

    # --------------------------------------------------------
    # Category summaries
    # --------------------------------------------------------

    base_categories, base_correct, base_total = summarize(
        base_scored
    )

    lora_categories, lora_correct, lora_total = summarize(
        lora_scored
    )

    print("=" * 70)
    print("CATEGORY RESULTS")
    print("=" * 70)

    category_order = [
        "Finance Concepts",
        "Finance Formulas",
        "Accounting Transactions",
        "Arithmetic",
        "Multi-Step Finance Reasoning",
    ]

    for category in category_order:

        b = base_categories[category]
        l = lora_categories[category]

        base_pct = (
            b["correct"] / b["total"] * 100
        )

        lora_pct = (
            l["correct"] / l["total"] * 100
        )

        print()
        print(category)

        print(
            f"  Base : {b['correct']}/{b['total']}"
            f" ({base_pct:.0f}%)"
        )

        print(
            f"  LoRA : {l['correct']}/{l['total']}"
            f" ({lora_pct:.0f}%)"
        )

        print(
            f"  Change: "
            f"{l['correct'] - b['correct']:+d}"
            f" question(s)"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    base_pct = base_correct / base_total * 100
    lora_pct = lora_correct / lora_total * 100

    print(
        f"Base : {base_correct}/{base_total}"
        f" ({base_pct:.0f}%)"
    )

    print(
        f"LoRA : {lora_correct}/{lora_total}"
        f" ({lora_pct:.0f}%)"
    )

    print(
        f"Change: "
        f"{lora_correct - base_correct:+d}"
        f" question(s)"
    )

    print(
        f"Percentage-point change: "
        f"{lora_pct - base_pct:+.1f} pp"
    )

    # --------------------------------------------------------
    # Per-question comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-QUESTION COMPARISON")
    print("=" * 70)

    for base_item in base_scored:

        example_id = base_item["id"]

        lora_item = next(
            item
            for item in lora_scored
            if item["id"] == example_id
        )

        base_mark = "PASS" if base_item["correct"] else "FAIL"
        lora_mark = "PASS" if lora_item["correct"] else "FAIL"

        change = ""

        if not base_item["correct"] and lora_item["correct"]:
            change = "  <-- LoRA IMPROVEMENT"

        elif base_item["correct"] and not lora_item["correct"]:
            change = "  <-- LoRA REGRESSION"

        print(
            f"{example_id:>3}   "
            f"Base={base_mark:<4}   "
            f"LoRA={lora_mark:<4}"
            f"{change}"
        )

    # --------------------------------------------------------
    # Save scoring results
    # --------------------------------------------------------

    output_file = (
        r"C:\FinMod\outputs"
        r"\finance_eval_v1_scored_comparison.json"
    )

    comparison = {
        "benchmark": "finance_eval_v1",
        "base": {
            "correct": base_correct,
            "total": base_total,
            "percentage": base_pct,
            "results": base_scored,
        },
        "lora": {
            "correct": lora_correct,
            "total": lora_total,
            "percentage": lora_pct,
            "results": lora_scored,
        },
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            comparison,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("Scoring complete")
    print("=" * 70)
    print()
    print("Saved:")
    print(output_file)


if __name__ == "__main__":
    main()