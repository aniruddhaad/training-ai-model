# FinMod Accountancy Semantic Evaluation v2

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
| Total | 458 | 458 |
| REVIEW | 251 | 378 |
| Scored | 207 | 80 |
| Correct | 0 | 0 |
| Partial | 9 | 12 |
| Incorrect | 189 | 68 |
| Unusable | 9 | 0 |
| Mean / 3 (scored only) | 1.000 | 1.150 |
| Correct rate (scored only) | 0.0% | 0.0% |
| Usable >=2 (scored only) | 4.3% | 15.0% |

## Comparable scored cases

- Both Base and LoRA received numeric scores on 75 cases.
- LoRA improved over Base on 1 cases.
- LoRA regressed on 0 cases.
- Scores were unchanged on 74 cases.

## Category breakdown

| source_class   | model   |   n_total |   review |   scored_n |   correct |   partial |   incorrect |   mean_scored |   correct_rate_scored |
|:---------------|:--------|----------:|---------:|-----------:|----------:|----------:|------------:|--------------:|----------------------:|
| Class11        | BASE    |       198 |      101 |         97 |         0 |         4 |          92 |         1.031 |                 0.000 |
| Class11        | LORA    |       198 |      159 |         39 |         0 |         4 |          35 |         1.103 |                 0.000 |
| Class12        | BASE    |       260 |      150 |        110 |         0 |         5 |          97 |         0.973 |                 0.000 |
| Class12        | LORA    |       260 |      219 |         41 |         0 |         8 |          33 |         1.195 |                 0.000 |

## Interpretation

`REVIEW` is intentional. It means the automated rules did not have enough
evidence to make a defensible correctness decision. Review cases must not be
counted as either correct or incorrect.

This report is therefore a more conservative benchmark than v1. It should be
used to identify objectively scored accounting/numerical behavior and to select
a smaller human-review set for conceptual cases.
