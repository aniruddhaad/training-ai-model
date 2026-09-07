# FinMod Gold Evaluation V1 — 100-Question Human Rubric

## Purpose

This is the project's stable, manually auditable evaluation set for comparing
Base vs LoRA accountancy behavior. It is selected from the 458 held-out
questions and is intended to remain fixed for future experiments.

## Scoring

Each answer is scored independently from 0–3.

### 3 — Correct
The answer gives the correct conclusion and, where the question requires it,
the correct calculation/accounting relationship. Minor wording differences do
not matter.

### 2 — Partially correct
The answer contains the main correct idea/result but is incomplete, misses an
important component, has a minor arithmetic/detail error, or gives a correct
result without sufficient reasoning when reasoning is explicitly requested.

### 1 — Incorrect
The answer is relevant to the question but gives a materially wrong accounting
concept, relationship, direction, calculation, or conclusion.

### 0 — Unusable
The answer is empty, refuses to answer, is unrelated, or is dominated by
prompt/dataset continuation without a usable answer to the actual question.

## Objective questions

For numerical/mechanism questions, prioritize:
1. Correct final numerical result.
2. Correct units/scaling.
3. Correct accounting direction/relationship.
4. Correct intermediate reasoning when requested.

A wrong final number with otherwise correct setup is normally 2 if the error is
minor and the reasoning is materially sound; otherwise 1.

## Accounting-state questions

Check the actual economic/accounting effect, not wording.

Examples of relationships to verify:
- Asset increase/decrease.
- Liability increase/decrease.
- Equity increase/decrease.
- Accounting equation remains balanced.
- Principal repayment reduces cash and liability, not equity.
- Owner investment increases assets and equity.
- Equipment purchased for cash changes asset composition but not total assets.
- AR collection changes AR to cash without changing total current assets.
- AP payment reduces both current assets and current liabilities.
- Working capital = Current Assets − Current Liabilities.
- A simultaneous equal decrease in CA and CL leaves working capital unchanged.

## Cash-flow questions

Check both the amount and mechanism where applicable:
- Non-cash depreciation is added back in the indirect method.
- Increase in receivables generally reduces operating cash flow relative to
  accounting profit.
- Credit revenue creates receivables rather than immediate cash.
- Working-capital adjustments must have the correct sign.

## Percentage / calculation questions

Recalculate independently. Do not accept a number merely because it appears
somewhere in the generated text.

Percentage increase:
    (New − Old) / Old × 100

Percentage decrease:
    (Old − New) / Old × 100

## Conceptual questions

Judge whether the explanation actually answers the asked accounting concept.
Do not require identical wording to the expected answer.

A response can be 3 even if it uses a different valid explanation.
A response should not be 3 merely because it shares keywords with the reference.

## Dataset-continuation rule

Only evaluate the answer to the current question. If the model begins another
`Question:` after answering, ignore everything after that point.

## Human-review protocol

For each Gold question:
1. Read the question.
2. Read the expected answer as the reference.
3. Read Base answer and assign 0–3.
4. Read LoRA answer and assign 0–3 independently.
5. Record a short note for any score below 3.
6. For disagreements between reviewers, discuss only the specific accounting
   criterion in dispute and record the final agreed score.

## Reporting

Report:
- Mean score / 3.
- Correct rate (score 3).
- Usable rate (score >= 2).
- Regression rate (LoRA < Base).
- Improvement rate (LoRA > Base).
- Category breakdown by benchmark_role, source_class, and difficulty.

Do not report automated heuristic scores as gold accuracy.
