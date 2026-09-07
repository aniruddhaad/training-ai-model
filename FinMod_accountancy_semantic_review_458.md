# FinMod — 458-Example Semantic Review Workbook

## Purpose

This is the **semantic-review stage** for the held-out accountancy validation set. It does **not** pretend that string similarity or numeric overlap is the same thing as accounting correctness.

The review file places, for every validation example:

- the question
- the source expected answer
- Base answer
- LoRA answer
- objective triage diagnostics
- blank semantic-score fields for final judgment

## Scoring rubric

Use these scores for both Base and LoRA:

- **3 = Correct** — accounting concept/reasoning and final answer are materially correct.
- **2 = Partially correct** — useful/correct core idea, but a material omission, incomplete reasoning, or minor numerical error.
- **1 = Incorrect** — materially wrong accounting treatment, calculation, or conclusion.
- **0 = Non-answer / unusable** — repetition, unrelated continuation, corrupted output, or no meaningful answer.

For each row, enter:

- `base_semantic_score`
- `lora_semantic_score`
- `base_error_notes`
- `lora_error_notes`

## Important limitation

`numeric_coverage` is only a **triage signal**. A model can contain the right numbers with the wrong accounting relationships, and a valid conceptual answer may not contain the same numbers as the reference. Therefore it must not be converted directly into an accuracy score.

## Current objective diagnostics

- Validation examples: **458**
- Base outputs containing a generated `Question:` continuation: **361**
- LoRA outputs containing a generated `Question:` continuation: **251**
- Mean Base numeric-value coverage: **0.110**
- Mean LoRA numeric-value coverage: **0.180**

These are diagnostic only, not final semantic scores.

## Recommended interpretation

After semantic scoring, calculate:

`accuracy = count(score == 3) / 458`

`partial_rate = count(score == 2) / 458`

`usable_rate = count(score >= 2) / 458`

`mean_score = mean(score)`

Then compare Base vs LoRA overall **and by topic/source class/difficulty**.

The final conclusion should be based on those semantic scores, not validation loss alone.
