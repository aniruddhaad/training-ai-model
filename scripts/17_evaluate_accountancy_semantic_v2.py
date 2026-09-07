"""
FinMod — Accountancy Semantic Evaluator v2

Goal:
    Re-score the existing 458 Base/LoRA generations without re-running the model.

Key corrections from v1:
    - Only the first generated answer is evaluated.
    - Any later "Question:" continuation is ignored.
    - Numeric evidence from later generated Q&A cannot affect the score.
    - Percentage calculations are checked explicitly.
    - Accounting-state relationships are checked explicitly where detectable.
    - Ambiguous conceptual cases are marked REVIEW instead of being forced into
      Correct/Incorrect by weak text similarity.

Input:
    accountancy_semantic_evaluation_v1.csv

Output:
    accountancy_semantic_evaluation_v2.csv
    accountancy_semantic_evaluation_v2.md
"""

from pathlib import Path
import re
import pandas as pd

INPUT = Path("/content/accountancy_semantic_evaluation_v1.csv")
OUT_DIR = Path("/content/drive/MyDrive/FinMod")

def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def answer_only(s):
    """Keep only the first model answer; discard generated dataset continuation."""
    s = str(s or "").strip()
    m = re.search(r"\n\s*question\s*:", s, flags=re.I)
    if m:
        s = s[:m.start()].strip()
    return s

def pct_values(s):
    return [float(m.group(1)) for m in
            re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", str(s or ""))]

def number_values(s):
    """
    Extract ordinary numbers and common financial units.
    This is deliberately conservative: dates are filtered where possible.
    """
    s = str(s or "").replace(",", "")
    out = []
    pat = r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(million|billion|thousand|lakh|crore|m|b|k)?"
    for m in re.finditer(pat, s, flags=re.I):
        x = float(m.group(1))
        unit = (m.group(2) or "").lower()
        # Ignore likely years.
        if unit == "" and 1900 <= abs(x) <= 2100 and float(x).is_integer():
            continue
        mult = {
            "million": 1e6, "m": 1e6,
            "billion": 1e9, "b": 1e9,
            "thousand": 1e3, "k": 1e3,
            "lakh": 1e5, "crore": 1e7
        }.get(unit, 1)
        out.append(x * mult)
    return out

def expected_percent(expected):
    ps = pct_values(expected)
    return ps[0] if ps else None

def close(a, b):
    return abs(a-b) <= max(0.01, abs(b)*0.002)

def contains_direction(text, positive, negative):
    t = norm(text)
    pos = any(x in t for x in positive)
    neg = any(x in t for x in negative)
    return pos, neg

def score_case(question, expected, answer):
    """
    Returns:
      score: 3 Correct / 2 Partial / 1 Incorrect / 0 Unusable / None REVIEW
      confidence: HIGH / MEDIUM / LOW
      reason
    """
    q = norm(question)
    e = norm(expected)
    a = answer_only(answer)

    if len(a) < 8:
        return 0, "HIGH", "empty/tiny first answer"

    # ------------------------------------------------------------
    # 1. Percentage questions — objective result check.
    # ------------------------------------------------------------
    if any(k in q for k in ["percentage increase", "percentage decrease",
                             "percent increase", "percent decrease",
                             "percentage change", "percent change",
                             "increase in percentage", "decrease in percentage"]):
        ep = expected_percent(expected)
        ap = pct_values(a)

        if ep is not None:
            if not ap:
                return 1, "HIGH", "no percentage result in first answer"
            if any(close(x, ep) for x in ap):
                # Correct result is enough for direct numerical percentage tasks.
                return 3, "HIGH", f"percentage result matches {ep:g}%"
            return 1, "HIGH", f"percentage result does not match expected {ep:g}%"

    # ------------------------------------------------------------
    # 2. Working-capital questions.
    # ------------------------------------------------------------
    if "working capital" in q:
        unchanged_expected = any(k in e for k in ["unchanged", "no change", "remains"])
        unchanged_answer = any(k in a.lower() for k in ["unchanged", "no change", "remains the same", "stays the same"])
        wrong_change = any(k in a.lower() for k in [
            "working capital decreases", "working capital decreased",
            "working capital increases", "working capital increased"
        ])

        if unchanged_expected:
            if unchanged_answer and not wrong_change:
                return 3, "HIGH", "working capital correctly identified as unchanged"
            if wrong_change:
                return 1, "HIGH", "working-capital direction contradicts expected answer"
            return 2, "MEDIUM", "working-capital conclusion not explicit enough"

        ev = number_values(expected)
        av = number_values(a)
        if ev:
            matches = sum(any(close(x,y) for y in av) for x in ev)
            if matches == len(ev):
                if unchanged_answer or any(k in a for k in ["current assets", "current liabilities"]):
                    return 3, "HIGH", "working-capital amount/mechanism supported"
                return 2, "MEDIUM", "working-capital amount matches but mechanism is incomplete"
            if matches:
                return 2, "MEDIUM", "some working-capital values match"
        return 1, "MEDIUM", "working-capital answer not supported"

    # ------------------------------------------------------------
    # 3. Cash-flow / depreciation / AR / AP questions.
    # ------------------------------------------------------------
    if any(k in q for k in [
        "operating cash flow", "cash flow", "depreciation",
        "accounts receivable", "account receivable", "receivables",
        "accounts payable", "account payable", "payable"
    ]):
        ev = number_values(expected)
        av = number_values(a)
        matches = sum(any(close(x,y) for y in av) for x in ev) if ev else 0

        mechanism_terms = [
            "add back", "added back", "non-cash", "does not affect cash",
            "no cash", "cash increases", "cash decreases",
            "receivable increases", "receivable decreases",
            "payable increases", "payable decreases",
            "subtract", "subtracted"
        ]
        mechanism = any(k in a for k in mechanism_terms)

        if ev and matches == len(ev) and mechanism:
            return 3, "HIGH", "cash-flow/accounting values and mechanism supported"
        if ev and matches == len(ev):
            return 2, "MEDIUM", "numerical result matches but mechanism is incomplete"
        if matches or mechanism:
            return 2, "MEDIUM", "cash-flow/accounting mechanism partially supported"
        return 1, "MEDIUM", "cash-flow/accounting answer not supported"

    # ------------------------------------------------------------
    # 4. Gross profit / margin / operating profit.
    # ------------------------------------------------------------
    if any(k in q for k in ["gross profit", "gross margin", "operating profit"]):
        ev = number_values(expected)
        av = number_values(a)
        ep = expected_percent(expected)
        ap = pct_values(a)

        num_ok = bool(ev) and sum(any(close(x,y) for y in av) for x in ev) == len(ev)
        pct_ok = ep is not None and any(close(x,ep) for x in ap)

        if (num_ok and (ep is None or pct_ok)) or (pct_ok and not ev):
            return 3, "HIGH", "profit/margin result matches expected"
        if num_ok or pct_ok:
            return 2, "MEDIUM", "some profit/margin result matches"
        return 1, "HIGH", "profit/margin result does not match"

    # ------------------------------------------------------------
    # 5. Accounting equation/state transitions.
    # ------------------------------------------------------------
    if any(k in q for k in [
        "accounting equation", "assets", "liabilities", "equity",
        "loan", "owner", "capital", "principal", "machinery",
        "equipment", "drawings", "draws"
    ]):
        ev = number_values(expected)
        av = number_values(a)
        num_matches = sum(any(close(x,y) for y in av) for x in ev) if ev else 0

        # Directional concepts.
        direction_pairs = [
            (["assets increase", "asset increase", "assets rise", "asset rises", "asset +"],
             ["assets decrease", "asset decrease", "assets fall", "asset falls", "asset -"]),
            (["liabilities increase", "liability increase", "liabilities rise", "liability rises", "liability +"],
             ["liabilities decrease", "liability decrease", "liabilities fall", "liability falls", "liability -"]),
            (["equity increase", "equity increases", "equity rise", "equity rises", "equity +"],
             ["equity decrease", "equity decreases", "equity fall", "equity falls", "equity -"])
        ]

        dir_hits = 0
        dir_conflicts = 0
        for pos_terms, neg_terms in direction_pairs:
            pos, neg = contains_direction(a, pos_terms, neg_terms)
            epos, eneg = contains_direction(e, pos_terms, neg_terms)
            if epos or eneg:
                if (epos and pos and not neg) or (eneg and neg and not pos):
                    dir_hits += 1
                elif (epos and neg) or (eneg and pos):
                    dir_conflicts += 1

        unchanged_expected = "unchanged" in e or "no change" in e
        unchanged_answer = any(k in a for k in ["unchanged", "no change", "remains unchanged", "remain unchanged"])

        if dir_conflicts:
            return 1, "HIGH", "accounting-state direction conflicts with expected answer"

        if ev and num_matches == len(ev) and (dir_hits or (unchanged_expected and unchanged_answer)):
            return 3, "HIGH", "accounting-state values and relationships supported"

        if dir_hits or (unchanged_expected and unchanged_answer) or num_matches:
            return 2, "MEDIUM", "accounting-state answer partially supported"

        # Don't force conceptual accounting questions into a false score.
        return None, "LOW", "conceptual accounting case requires review"

    # ------------------------------------------------------------
    # 6. Generic conceptual cases.
    # ------------------------------------------------------------
    # Deliberately conservative phrase overlap. It can support triage, but
    # never awards Correct by itself.
    et = set(re.findall(r"\b[a-z]{4,}\b", e))
    at = set(re.findall(r"\b[a-z]{4,}\b", norm(a)))
    overlap = len(et & at) / max(1, len(et))

    if overlap >= 0.55:
        return None, "LOW", "strong conceptual overlap; human review required"
    if overlap >= 0.25:
        return None, "LOW", "partial conceptual overlap; human review required"
    return 1, "LOW", "weak conceptual alignment"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")

    df = pd.read_csv(INPUT)
    required = {
        "validation_index", "source_class", "topic", "question",
        "expected_answer", "base_answer", "lora_answer"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if len(df) != 458:
        raise RuntimeError(f"Expected 458 rows, got {len(df)}")

    rows = []
    for _, r in df.iterrows():
        bs, bc, br = score_case(r["question"], r["expected_answer"], r["base_answer"])
        ls, lc, lr = score_case(r["question"], r["expected_answer"], r["lora_answer"])

        # Numeric score for aggregate analysis: REVIEW is excluded from
        # correctness denominators and represented as -1 in this internal field.
        rows.append({
            "validation_index": r["validation_index"],
            "source_class": r["source_class"],
            "topic": r["topic"],
            "difficulty": r.get("difficulty",""),
            "question": r["question"],
            "expected_answer": r["expected_answer"],
            "base_answer_first": answer_only(r["base_answer"]),
            "lora_answer_first": answer_only(r["lora_answer"]),
            "base_score": bs if bs is not None else "REVIEW",
            "lora_score": ls if ls is not None else "REVIEW",
            "base_confidence": bc,
            "lora_confidence": lc,
            "base_reason": br,
            "lora_reason": lr,
        })

    out = pd.DataFrame(rows)

    def stats(col):
        s = out[col]
        scored = pd.to_numeric(s, errors="coerce").dropna()
        return {
            "n_total": len(s),
            "review": int(s.eq("REVIEW").sum()),
            "correct": int((s == 3).sum()),
            "partial": int((s == 2).sum()),
            "incorrect": int((s == 1).sum()),
            "unusable": int((s == 0).sum()),
            "scored_n": len(scored),
            "mean_scored": float(scored.mean()) if len(scored) else float("nan"),
            "correct_rate_scored": float((scored == 3).mean()) if len(scored) else float("nan"),
            "usable_rate_scored": float((scored >= 2).mean()) if len(scored) else float("nan"),
        }

    sb, sl = stats("base_score"), stats("lora_score")

    # Objective disagreement groups, excluding REVIEW-vs-REVIEW.
    def numeric(x):
        try: return int(x)
        except: return None

    changes = []
    for _, r in out.iterrows():
        b, l = numeric(r["base_score"]), numeric(r["lora_score"])
        if b is not None and l is not None:
            changes.append(l-b)
    improved = sum(x > 0 for x in changes)
    regressed = sum(x < 0 for x in changes)
    same = sum(x == 0 for x in changes)

    cat_rows = []
    for cls, g in out.groupby("source_class"):
        for model in ["base_score", "lora_score"]:
            s = pd.to_numeric(g[model], errors="coerce").dropna()
            cat_rows.append({
                "source_class": cls,
                "model": model.replace("_score","").upper(),
                "n_total": len(g),
                "review": int(g[model].eq("REVIEW").sum()),
                "scored_n": len(s),
                "correct": int((s==3).sum()),
                "partial": int((s==2).sum()),
                "incorrect": int((s==1).sum()),
                "mean_scored": float(s.mean()) if len(s) else float("nan"),
                "correct_rate_scored": float((s==3).mean()) if len(s) else float("nan"),
            })
    cat = pd.DataFrame(cat_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "accountancy_semantic_evaluation_v2.csv"
    md_path = OUT_DIR / "accountancy_semantic_evaluation_v2.md"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = f"""# FinMod Accountancy Semantic Evaluation v2

## Purpose

This version re-scores the existing 458 Base/LoRA generations without running
the model again. It fixes the v1 continuation-leak problem.

## Key methodological corrections

- Only the first generated answer is evaluated.
- Any later generated `Question:` continuation is discarded.
- Numeric evidence from later generated Q&A cannot influence scoring.
- Direct percentage questions are checked against the expected percentage.
- Accounting-state and working-capital relationships are checked explicitly where detectable.
- Ambiguous conceptual cases are marked `REVIEW` instead of being forced into a
  potentially misleading accuracy score.

## Overall

| Metric | Base | LoRA |
|---|---:|---:|
| Total | {sb["n_total"]} | {sl["n_total"]} |
| REVIEW | {sb["review"]} | {sl["review"]} |
| Scored | {sb["scored_n"]} | {sl["scored_n"]} |
| Correct | {sb["correct"]} | {sl["correct"]} |
| Partial | {sb["partial"]} | {sl["partial"]} |
| Incorrect | {sb["incorrect"]} | {sl["incorrect"]} |
| Unusable | {sb["unusable"]} | {sl["unusable"]} |
| Mean / 3 (scored only) | {sb["mean_scored"]:.3f} | {sl["mean_scored"]:.3f} |
| Correct rate (scored only) | {sb["correct_rate_scored"]:.1%} | {sl["correct_rate_scored"]:.1%} |
| Usable >=2 (scored only) | {sb["usable_rate_scored"]:.1%} | {sl["usable_rate_scored"]:.1%} |

## Comparable scored cases

- Both Base and LoRA received numeric scores on {len(changes)} cases.
- LoRA improved over Base on {improved} cases.
- LoRA regressed on {regressed} cases.
- Scores were unchanged on {same} cases.

## Category breakdown

{cat.to_markdown(index=False, floatfmt=".3f")}

## Interpretation

`REVIEW` is intentional. It means the automated rules did not have enough
evidence to make a defensible correctness decision. Review cases must not be
counted as either correct or incorrect.

This report is therefore a more conservative benchmark than v1. It should be
used to identify objectively scored accounting/numerical behavior and to select
a smaller human-review set for conceptual cases.
"""
    md_path.write_text(report, encoding="utf-8")

    print("=== FinMod Accountancy Semantic Evaluation v2 ===")
    print(pd.DataFrame([sb, sl], index=["Base","LoRA"]).to_string())
    print(f"\nComparable scored cases: {len(changes)}")
    print(f"LoRA improved: {improved}")
    print(f"LoRA regressed: {regressed}")
    print(f"Unchanged: {same}")
    print("\nCategory breakdown:")
    print(cat.to_string(index=False))
    print("\nSaved:", csv_path)
    print("Saved:", md_path)

if __name__ == "__main__":
    main()
