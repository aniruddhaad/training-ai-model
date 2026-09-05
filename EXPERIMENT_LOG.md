# Finance/Accounting LLM Experiment Log

Living project notebook for the `Training AI Model` finance/accounting LLM project.

This log records controlled experiments, diagnostics, benchmark design decisions, and lessons learned. Future experiments should be appended here rather than reconstructed from chat history.

## Experiment Index

| ID | Artifact / Adapter | Main Change | Training Examples | Final Training Loss | Evaluation Summary | Status |
|---|---|---|---:|---:|---|---|
| Exp1 | `finance_lora_r8_qv` | First LoRA train, r=8 on Q/V, full-sequence loss | 29 | 0.9442 | Improved several finance answers, but arithmetic and reasoning remained weak | Complete |
| Exp2 | `finance_lora_r8_qv_response_only` | Same as Exp1, but response-only loss | 29 | 0.7641 | Cleaner objective and formatting; arithmetic/formula failures persisted | Complete |
| Exp3 | `finance_lora_r8_qkvo_response_only` | Same data/objective, target Q/K/V/O | 29 | 0.6072 | Lower loss and some concept improvements; Q3/Q6 still failed | Complete |
| Exp4 | `finance_lora_r8_qkvo_response_only_v2` | Added targeted v2 data | 46 after exclusions | 0.5550 | Gross margin fixed; percentage formula improved but arithmetic wrong | Complete |
| Exp5 | `finance_lora_r8_qkvo_response_only_v3` | Added v3 numerical/accounting reasoning examples | 54 after exclusions | 0.5092 | Training fit improved; later eval showed Q5 still not fully fixed | Complete |
| Exp6 | `finance_lora_r8_qkvo_response_only_v4` | Added v4 explicit arithmetic/accounting examples | 64 after exclusions | 0.5142 | Bank-loan accounting fixed; Q3 still 10%; some regressions | Complete |
| Exp7 | `finance_lora_r16_qkvo_response_only_v4` | Doubled LoRA rank from 8 to 16 | 64 after exclusions | 0.4370 | Lower loss; same 7-question eval still failed Q3 | Complete |
| Diagnostic A | `04_arithmetic_diagnostic.py` | Base vs r16 arithmetic/memorization test | N/A | N/A | Showed arithmetic/composition is unreliable; Q3 contamination must be flagged | Complete |
| Diagnostic B | structured reasoning diagnostic | Tested whether step-by-step prompting fixes failures | N/A | N/A | Helped identify composition/causal consistency as weak points | Complete |
| Capability Eval | `06_capability_eval.py` | Broader manual capability profile | N/A | N/A | r16 showed strongest performance on formulas/arithmetic; accounting and multi-step reasoning remained mixed, with manual scoring inconsistency noted | Complete |
| Benchmark v1 | `finance_eval_v1.jsonl` | Fixed 25-question held-out benchmark | N/A | N/A | 5 categories x 5 questions; must never be trained on | Complete |
| Exp8 | `finance_eval_v1_r16_qkvo_v4_results.jsonl` | Ran fixed benchmark v1 on current best LoRA | N/A | N/A | 14/25 overall | Complete |
| Exp9 | `finance_eval_v1_base_results.jsonl` | Ran fixed benchmark v1 on base model control | N/A | N/A | 12/25 overall; LoRA improved +2 questions | Complete |

## Current Project State

### Current Best Adapter

Current best adapter by fixed benchmark v1:

```text
C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4
```

It is the best measured LoRA adapter so far, with 14/25 on benchmark v1 versus 12/25 for the base model. It should be treated as the current best training-only adapter, not as a production-quality finance model.

### Important Contamination Warning

The exact percentage-increase question:

```text
Revenue increased from $2 million to $2.4 million. What is the percentage increase?
```

was included in the v4 training data. Therefore, any direct result on that exact question is not valid evidence of generalization. This is especially important because the model still answered `10%` in several runs even after exposure, showing both contamination risk and arithmetic unreliability.

The fixed benchmark v1 was later created with different wording/numbers where possible. It must remain separate from training data.

## Environment

Recorded from the project conversation:

```text
OS / shell: Windows PowerShell
Project root: C:\FinMod
Python environment: C:\FinMod\.venv
Base model: C:\FinMod\models\SmolLM2-360M
Training/eval device: CPU
CUDA availability: torch.cuda.is_available() == False
GPU observed: NVIDIA GT 710 present, not used for PyTorch acceleration
RAM observed during training: about 12.6-12.9 GB used of 15.8 GB, about 80-82%
CPU observed during training: about 52-77%, boosting around 3.68-3.78 GHz
Libraries used: transformers, peft, torch
Generation for evaluations: deterministic, do_sample=False
```

### Directory Structure

```text
C:\FinMod\
  .venv\
  models\
    SmolLM2-360M\
  data\
    finance_tiny.jsonl
    finance_tiny_v2.jsonl
    finance_tiny_v3.jsonl
    finance_tiny_v4.jsonl
    finance_eval_v1.jsonl
  scripts\
    01_attach_lora.py
    02_train_lora.py
    03_evaluate_lora.py
    04_arithmetic_diagnostic.py
    06_capability_eval.py
    07_score_capability.py
    08_run_eval_v1.py
    09_run_base_eval_v1.py
  adapters\
    finance_lora_r8_qv
    finance_lora_r8_qv_response_only
    finance_lora_r8_qkvo_response_only
    finance_lora_r8_qkvo_response_only_v2
    finance_lora_r8_qkvo_response_only_v3
    finance_lora_r8_qkvo_response_only_v4
    finance_lora_r16_qkvo_response_only_v4
  outputs\
    finance_eval_v1_r16_qkvo_v4_results.jsonl
    finance_eval_v1_base_results.jsonl
```

## Benchmark Methodology

### Early Seven-Question Evaluation

The first evaluation script compared the base model against a selected LoRA adapter on the same seven finance/accounting questions:

1. Difference between gross profit and net profit.
2. Working capital from current assets of $200,000 and current liabilities of $125,000.
3. Percentage increase from $2 million to $2.4 million.
4. Why an increase in accounts receivable can reduce operating cash flow.
5. Accounting equation impact of a $50,000 bank loan.
6. Gross margin from revenue of $1,000,000 and gross profit of $400,000.
7. Why depreciation is added back under the indirect method.

The evaluation was deterministic and manually judged. The script was later improved to decode only newly generated tokens, limit generation length, and remove leaked prompt markers such as `### Question:`.

### Fixed Benchmark v1

`C:\FinMod\data\finance_eval_v1.jsonl` is the fixed held-out benchmark. It contains 25 examples across five categories:

| Category | Count |
|---|---:|
| Finance Concepts | 5 |
| Finance Formulas | 5 |
| Accounting Transactions | 5 |
| Arithmetic | 5 |
| Multi-Step Finance Reasoning | 5 |

This benchmark is now the stable comparison set for base model, LoRA adapters, and future tool-assisted architectures. Do not overwrite, retrain on, or casually edit it.

## Dataset Notes

### `finance_tiny.jsonl`

Initial 29-example training dataset. It used instruction/response JSONL records and covered:

- Basic accounting and finance definitions: revenue, asset, liability, equity.
- Accounting equation.
- Income statement, balance sheet, cash flow.
- Gross profit, operating profit, net profit.
- Accounts receivable, accounts payable.
- Working capital.
- Simple profit and margin calculations.
- Depreciation and cash flow.

### `finance_tiny_v2.jsonl`

18 targeted examples added after Exp3. Designed to improve formula coverage and finance reasoning. It included varied percentage-increase examples such as:

- $1M -> $1.1M = 10%
- $500K -> $600K = 20%
- $4M -> $5M = 25%

The gross-margin evaluation question was found in the dataset but was excluded from training by script. The `$2M -> $2.4M` Q3 evaluation question was not in v1/v2 at that time.

### `finance_tiny_v3.jsonl`

8 additional examples focused on:

- Explicit percentage calculations.
- Accounting-equation transaction cases.
- Contrasting bank loans, owner investment, loan repayment, equipment purchase, and cash revenue.

The bank-loan training example was near the evaluation question but not exact.

### `finance_tiny_v4.jsonl`

10 additional examples focused on highly explicit arithmetic and accounting reasoning. The combined source dataset became:

```text
29 original
+18 v2
+ 8 v3
+10 v4
=65 total source examples
- 1 excluded exact evaluation example
=64 final training examples
```

Important: v4 included the exact `$2M -> $2.4M` percentage-increase question, creating a contamination issue for that diagnostic question.

## Experiments

### Exp1 - First LoRA Training Run

Objective / hypothesis:

Test whether a tiny LoRA adapter can train successfully on the local Windows CPU setup and shift SmolLM2-360M toward finance/accounting responses.

Model / configuration:

```text
Base model: SmolLM2-360M
Model path: C:\FinMod\models\SmolLM2-360M
LoRA rank: r=8
LoRA alpha: 16
Target modules: q_proj, v_proj
Dropout: 0.05
Bias: none
Task type: CAUSAL_LM
Device: CPU
Max length: 128
Batch size: 1
Epochs: 3
Learning rate: 2e-4
Trainable params: 819,200
Trainable percent: 0.2259%
Output: C:\FinMod\adapters\finance_lora_r8_qv
```

Dataset:

`finance_tiny.jsonl`, 29 examples.

Loss objective:

Full sequence loss. The model was trained on the formatted instruction and response text, not only the response.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.8550 |
| 2 | 1.2769 |
| 3 | 0.9442 |

Evaluation findings:

- Improved Q1 gross vs net profit compared with base, though not perfect.
- Improved Q2 working-capital application; produced $75,000.
- Q3 percentage increase remained wrong at 10% instead of 20%.
- Q4 accounts receivable explanation used the wrong mechanism.
- Q5 bank loan/accounting equation improved only partially.
- Q6 gross margin regressed badly, answering `$600,000` instead of 40%.
- Q7 depreciation explanation remained correct.

Interpretation:

The adapter learned the tiny dataset and changed model behavior, but falling training loss did not prove generalization. The model picked up style and some finance concepts while still failing arithmetic and causal reasoning.

Lessons learned:

- LoRA training worked on CPU.
- The base model weights stayed frozen while LoRA weights were trained.
- Full-sequence loss spent training signal on reproducing the prompt format.
- A falling training loss can coexist with wrong answers on unseen questions.

### Exp2 - Response-Only Loss

Objective / hypothesis:

Keep model, rank, target modules, dataset, learning rate, epochs, and max length fixed, but train only on response tokens. Hypothesis: a cleaner instruction-tuning objective should improve answer behavior.

Model / configuration:

```text
Base model: SmolLM2-360M
LoRA rank: r=8
LoRA alpha: 16
Target modules: q_proj, v_proj
Dropout: 0.05
Epochs: 3
Learning rate: 2e-4
Max length: 128
Trainable params: 819,200
Trainable percent: 0.2259%
Output: C:\FinMod\adapters\finance_lora_r8_qv_response_only
```

Dataset:

`finance_tiny.jsonl`, 29 examples.

Exclusions:

None recorded for this run.

Training settings:

Response-only labels: instruction tokens set to `-100`; response tokens contributed to loss.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.2122 |
| 2 | 0.9576 |
| 3 | 0.7641 |

Evaluation findings:

- Q1 improved to a cleaner finance answer.
- Q2 remained good and calculated $75,000.
- Q3 remained 100%, still wrong.
- Q4 remained wrong, saying cash must be paid out to settle accounts receivable.
- Q5 gave the high-level idea that the equation remains unchanged but lacked mechanics.
- Q6 regressed versus base, giving a dollar amount instead of 40%.
- Q7 remained correct.

Interpretation:

Response-only loss is a better objective for instruction tuning and improved formatting/completeness, but it did not fix generalization, arithmetic, formula selection, or causal explanation.

Lessons learned:

- Better loss masking does not automatically create better reasoning.
- The model learned some answer style and simple applications.
- Arithmetic/generalization needed separate diagnosis.

### Exp3 - Broader LoRA Target Modules

Objective / hypothesis:

Test whether restricting adaptation to Q/V was limiting learning. Change target modules to Q/K/V/O while keeping r=8 and response-only loss.

Model / configuration:

```text
Base model: SmolLM2-360M
LoRA rank: r=8
LoRA alpha: 16
Target modules: q_proj, k_proj, v_proj, o_proj
Dropout: 0.05
Epochs: 3
Learning rate: 2e-4
Max length: 128
Trainable params: 1,638,400
Trainable percent: 0.4508%
Output: C:\FinMod\adapters\finance_lora_r8_qkvo_response_only
```

Dataset:

`finance_tiny.jsonl`, 29 examples.

Exclusions:

None recorded for this run.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.1671 |
| 2 | 0.8516 |
| 3 | 0.6072 |

Evaluation findings:

- Q1 became cleaner.
- Q2 remained correct.
- Q3 remained 100%, wrong.
- Q4 improved compared with Exp2, closer to the AR mechanism.
- Q5 said the accounting equation remains unchanged but invented a purchase-of-asset scenario.
- Q6 still failed, giving `$600,000`.
- Q7 remained correct.

Interpretation:

Broader target placement increased trainable parameters and fit the training responses better. It improved some conceptual behavior, but repeated the same arithmetic/formula failures.

Lessons learned:

- Parameter count and target placement both matter.
- r=8 QKVO has the same trainable parameter count as r=16 QV would have, but distributes capacity differently.
- Q3 and Q6 failures survived multiple configurations, pointing toward data/design/capability issues rather than only LoRA placement.

### Exp4 - Add Targeted v2 Data

Objective / hypothesis:

Improve formula and finance reasoning by adding targeted examples, while keeping the Exp3 LoRA configuration.

Model / configuration:

```text
Base model: SmolLM2-360M
LoRA rank: r=8
Target modules: q_proj, k_proj, v_proj, o_proj
Objective: response-only loss
Epochs: 3
Learning rate: 2e-4
Trainable params: 1,638,400
Output: C:\FinMod\adapters\finance_lora_r8_qkvo_response_only_v2
```

Dataset:

```text
Dataset 1: 29 examples
Dataset 2: 18 examples
Combined: 47 examples
Excluded exact evaluation examples: 1
Final training dataset: 46 examples
```

Exclusions:

The exact gross-margin evaluation example was excluded:

```text
A company has revenue of $1,000,000 and gross profit of $400,000. What is its gross margin?
```

At this point, the `$2M -> $2.4M` percentage-increase evaluation question was not duplicated in training.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.1009 |
| 2 | 0.7757 |
| 3 | 0.5550 |

Evaluation findings:

- Q1 correct.
- Q2 correct.
- Q3 changed from 100% to a correct-looking formula but wrong final answer: 10%.
- Q4 became correct or much closer to correct.
- Q5 still wrong; equity handling remained unreliable.
- Q6 fixed: output became 40%.
- Q7 correct.

Interpretation:

Targeted data helped. The gross-margin result was the strongest evidence because the exact evaluation question was excluded from training. Q3 became more informative: the model learned the formula shape but not reliable arithmetic execution.

Lessons learned:

- Training data design can change behavior meaningfully.
- Formula knowledge and numerical execution are separable.
- Contrasting accounting transaction examples were needed.

### Exp5 - Add v3 Reasoning and Accounting Examples

Objective / hypothesis:

Keep r=8 QKVO response-only fixed and add explicit examples for numerical reasoning and accounting-equation transaction distinctions.

Model / configuration:

```text
Base model: SmolLM2-360M
LoRA rank: r=8
Target modules: q_proj, k_proj, v_proj, o_proj
Objective: response-only loss
Epochs: 3
Learning rate: 2e-4
Trainable params: 1,638,400
Trainable percent: 0.4508%
Output: C:\FinMod\adapters\finance_lora_r8_qkvo_response_only_v3
```

Dataset:

```text
Dataset 1: 29 examples
Dataset 2: 18 examples
Dataset 3: 8 examples
Combined: 55 examples
Excluded exact evaluation examples: 1
Final training dataset: 54 examples
```

Exclusions:

The gross-margin exact evaluation example remained excluded. The bank-loan example was near the evaluation question but not exact.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.0791 |
| 2 | 0.7336 |
| 3 | 0.5092 |

Evaluation findings recorded later in comparison:

- Q1 maintained.
- Q2 maintained.
- Q3 remained 10%, not 20%.
- Q4 improved but retained contradictory wording.
- Q5 still incorrectly increased equity in the bank-loan case.
- Q6 remained correct at 40%.
- Q7 remained correct in Exp5 comparison.

Interpretation:

Additional examples lowered training loss and improved/maintained several behaviors, but still did not solve percentage arithmetic or robust accounting transaction reasoning.

Lessons learned:

- More targeted examples can help without guaranteeing reliability.
- The model tends to learn local patterns and associations before robust procedures.
- Need to distinguish knowledge adaptation from computation.

### Exp6 - Add v4 Explicit Arithmetic/Accounting Examples

Objective / hypothesis:

Add more explicit arithmetic and accounting examples to see whether carefully structured data can fix the stubborn Q3 and Q5 failures while preserving Q6.

Model / configuration:

```text
Base model: SmolLM2-360M
LoRA rank: r=8
Target modules: q_proj, k_proj, v_proj, o_proj
Objective: response-only loss
Epochs: 3
Learning rate: 2e-4
Trainable params: 1,638,400
Trainable percent: 0.4508%
Output: C:\FinMod\adapters\finance_lora_r8_qkvo_response_only_v4
```

Dataset:

```text
Dataset 1: 29 examples
Dataset 2: 18 examples
Dataset 3: 8 examples
Dataset 4: 10 examples
Combined: 65 examples
Excluded exact evaluation examples: 1
Final training dataset: 64 examples
```

Exclusions:

One exact evaluation example excluded. Important: the exact `$2M -> $2.4M` percentage-increase question was included in v4 training, so Q3 was no longer a clean generalization test.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.0653 |
| 2 | 0.7254 |
| 3 | 0.5142 |

Evaluation findings:

- Q1 maintained.
- Q2 maintained.
- Q3 still answered 10%, even after contamination.
- Q4 remained flawed or contradictory.
- Q5 fixed: assets increase, liabilities increase, equity does not change.
- Q6 maintained at 40%.
- Q7 regressed conceptually, saying depreciation is not included in operating cash flow rather than clearly explaining the indirect-method add-back.

Interpretation:

Exp6 provided real evidence that targeted accounting transaction data helped Q5. But the model still failed Q3 even when the exact example had been seen, suggesting arithmetic/composition is a deeper limitation.

Lessons learned:

- Targeted data can fix specific accounting transaction behavior.
- Adding examples can regress other concepts.
- Q3 is contaminated and must not be used as clean generalization evidence.
- The model is not a reliable calculator.

### Exp7 - Double LoRA Rank to r=16

Objective / hypothesis:

Test whether doubling LoRA capacity improves remaining failures while holding dataset, target modules, objective, learning rate, epochs, and max length fixed.

Model / configuration:

```text
Base model: SmolLM2-360M
LoRA rank: r=16
LoRA alpha: 32
Target modules: q_proj, k_proj, v_proj, o_proj
Dropout: 0.05
Objective: response-only loss
Epochs: 3
Learning rate: 2e-4
Max length: 128
Trainable params: 3,276,800
All params: 365,097,920
Trainable percent: 0.8975%
Output: C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4
```

Dataset:

Same 64-example final dataset as Exp6.

Results:

| Epoch | Average Loss |
|---:|---:|
| 1 | 1.0190 |
| 2 | 0.6557 |
| 3 | 0.4370 |

Evaluation findings:

- Q1 maintained.
- Q2 maintained.
- Q3 still 10%, no improvement.
- Q4 still problematic.
- Q5 maintained correct bank-loan accounting.
- Q6 maintained correct gross margin.
- Q7 improved back to a good depreciation add-back answer.

Interpretation:

r=16 fit the training data better than r=8, but the stubborn arithmetic failure did not improve. The bottleneck was not simply adapter capacity.

Lessons learned:

- Capacity improves training loss but not necessarily reasoning.
- Arithmetic/composition failure survived response-only loss, QKVO targeting, explicit data, and doubled rank.
- Future work should focus on evaluation rigor, data design, deterministic tools, or stronger base models rather than blindly increasing rank.

## Diagnostics

### Arithmetic Diagnostic

Objective:

Determine whether the Q3 percentage failure came from missing finance knowledge, bad arithmetic, poor multi-step composition, or memorization/generalization failure.

Models compared:

- Base `SmolLM2-360M`
- `SmolLM2-360M + finance_lora_r16_qkvo_response_only_v4`

Questions included:

```text
What is 2.4 - 2.0?
What is 0.4 / 2.0?
What is 0.2 x 100?
What is (2.4 - 2.0) / 2.0 x 100?
Revenue increased from $2 million to $2.4 million. Calculate the percentage increase step by step.
Revenue increased from $3 million to $3.6 million. What is the percentage increase?
Revenue increased from $10 million to $12 million. What is the percentage increase?
Revenue increased from $800,000 to $1,000,000. What is the percentage increase?
```

Key interpretation:

The diagnostic was designed to distinguish isolated arithmetic from composed finance arithmetic and to test memorization versus generalization. The exact `$2M -> $2.4M` question had been included in v4 training, so that direct item is contaminated.

Lesson:

The model can know a formula and still compute the final answer incorrectly.

### Structured Reasoning Diagnostic

Objective:

Test whether explicit step-by-step prompting or structured decomposition improves the known weak areas: percentage arithmetic, accounts receivable cash-flow reasoning, and accounting-equation state transitions.

Findings:

The project conclusion from the diagnostics was that structured reasoning helps expose where the model fails, but it does not make SmolLM2-360M a deterministic calculator or validator. It can produce correct-looking intermediate work with incorrect final arithmetic.

Lesson:

For finance use, the architecture should eventually combine the LLM with deterministic calculation and accounting validation tools.

## Capability Evaluation

Script:

```text
C:\FinMod\scripts\06_capability_eval.py
```

Model:

```text
SmolLM2-360M + finance_lora_r16_qkvo_response_only_v4
```

Categories:

- Finance Concepts
- Finance Formulas
- Accounting Transactions
- Arithmetic
- Multi-Step Reasoning

Manual capability profile from that run:

| Area | Approx Result | Assessment |
|---|---:|---|
| Basic finance concepts | about 3/5 | Mixed |
| Finance formulas | 4/5 | Strongest area |
| Accounting transactions | 2/5 | Weak/mixed |
| Simple arithmetic | 5/6 | Surprisingly good but unreliable |
| Multi-step finance | inconsistent manual notes | Weak/mixed |

Note: the conversation contains an inconsistency for the multi-step capability diagnostic. A summary table described multi-step finance as 2/5, while later item-by-item commentary treated several E items as correct. Treat this diagnostic as qualitative only; use benchmark v1 for formal scoring.

Important findings:

- The model can produce finance-sounding explanations that are internally inconsistent.
- Asset/liability definitions can bleed into each other.
- Percentage calculations and formula execution are not reliable.
- Accounting state transitions need structured representation.
- The model knows many ingredients but struggles to reliably combine concepts, numbers, and relationships.

## Fixed Benchmark v1 Results

### Exp8 - LoRA r16 QKVO v4 on Benchmark v1

Script:

```text
C:\FinMod\scripts\08_run_eval_v1.py
```

Model:

```text
Base: C:\FinMod\models\SmolLM2-360M
Adapter: C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4
Eval file: C:\FinMod\data\finance_eval_v1.jsonl
Output: C:\FinMod\outputs\finance_eval_v1_r16_qkvo_v4_results.jsonl
```

Generation:

```text
max_new_tokens = 60
do_sample = False
```

Score:

| Category | LoRA r16 QKVO v4 |
|---|---:|
| Finance Concepts | 4/5 |
| Finance Formulas | 4/5 |
| Accounting Transactions | 2/5 |
| Arithmetic | 3/5 |
| Multi-Step Finance Reasoning | 1/5 |
| Total | 14/25, 56% |

Interpretation:

LoRA improved finance formulas, accounting transactions, and response behavior, but did not solve arithmetic or multi-step reasoning.

### Exp9 - Base Model Control on Benchmark v1

Script:

```text
C:\FinMod\scripts\09_run_base_eval_v1.py
```

Model:

```text
Base: C:\FinMod\models\SmolLM2-360M
Adapter: none
Eval file: C:\FinMod\data\finance_eval_v1.jsonl
Output: C:\FinMod\outputs\finance_eval_v1_base_results.jsonl
```

Generation:

```text
max_new_tokens = 60
do_sample = False
```

Score:

| Category | Base Model |
|---|---:|
| Finance Concepts | 4/5 |
| Finance Formulas | 3/5 |
| Accounting Transactions | 1/5 |
| Arithmetic | 4/5 |
| Multi-Step Finance Reasoning | 0/5 |
| Total | 12/25, 48% |

Base vs LoRA:

| Category | Base | LoRA r16 QKVO v4 | Change |
|---|---:|---:|---:|
| Finance Concepts | 4/5 | 4/5 | 0 |
| Finance Formulas | 3/5 | 4/5 | +1 |
| Accounting Transactions | 1/5 | 2/5 | +1 |
| Arithmetic | 4/5 | 3/5 | -1 |
| Multi-Step Finance Reasoning | 0/5 | 1/5 | +1 |
| Total | 12/25 | 14/25 | +2 |

Interpretation:

The LoRA adapter produced a real but modest improvement: +2 questions, +8 percentage points. It did not make the model better at everything. Arithmetic became worse on this benchmark, while finance-specific formula and accounting behavior improved.

## Main Lessons So Far

1. LoRA works: the adapter changes behavior and improves selected finance responses.
2. Response-only loss is cleaner than full-sequence loss, but it does not guarantee better reasoning.
3. Targeting Q/K/V/O improves training fit and affects concepts, but does not by itself solve arithmetic.
4. Better targeted data helped more than purely changing rank.
5. More LoRA capacity lowered training loss but did not fix the stubborn percentage arithmetic failure.
6. Training can improve one capability while regressing another.
7. The model often knows formulas but does not reliably execute them.
8. Accounts receivable and accounting-equation cases reveal causal/state-transition weaknesses.
9. Fixed held-out benchmarks are essential; otherwise it is too easy to mistake memorization for generalization.
10. The likely next major architecture is LLM plus deterministic tools: calculator, accounting equation validator, and structured transaction representation.

## Next Planned Steps

Immediate next step:

Create a scoring/comparison script that automatically reads benchmark result JSONL files and produces category-level scores for:

```text
Finance Concepts
Finance Formulas
Accounting Transactions
Arithmetic
Multi-Step Finance Reasoning
Total
```

Recommended next experiment direction:

- Do not train another LoRA immediately.
- First make benchmark scoring more rigorous and repeatable.
- Then test a focused hypothesis around training-data design for accounting state transitions and multi-step finance reasoning.
- Consider tool-assisted calculation and validation, because the 360M model should not be expected to behave like a calculator.

Possible future experiment:

```text
Can examples specifically designed around accounting state transitions and multi-step reasoning improve C/E categories without damaging arithmetic?
```

Future architecture idea:

```text
User question
  -> LLM identifies finance task
  -> deterministic calculator or accounting validator handles exact computation/state checks
  -> LLM explains the verified result
```

## Logging Rules Going Forward

For every future experiment, append:

- Experiment ID and short name.
- Objective / hypothesis.
- Model and adapter configuration.
- Dataset versions and exact counts.
- Exclusions and contamination notes.
- Training settings.
- Training loss curve.
- Evaluation script and benchmark used.
- Raw result file path.
- Category scores.
- Interpretation.
- Lessons learned.
- Next decision.

Do not edit `C:\FinMod\data\finance_eval_v1.jsonl` unless deliberately creating a new benchmark version such as `finance_eval_v2.jsonl`.
