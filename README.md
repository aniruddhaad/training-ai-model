# FinMod — Finance & Accounting LLM

**FinMod** is an experimental research project focused on learning how to adapt a small open-weight language model into a more useful finance and accounting model.

The project is deliberately built around **small, controlled experiments on local hardware** rather than starting with a large model, a huge dataset, or an expensive training environment.

The goal is not simply to produce a model that appears to answer finance questions.

The goal is to understand:

- what a small language model already knows,
- what fine-tuning can actually improve,
- where fine-tuning fails,
- how dataset design affects behavior,
- how to evaluate numerical and accounting reasoning,
- how to detect training-data contamination,
- and where deterministic tools, retrieval, and external financial data are needed instead of relying entirely on the language model.

---

## Project Status

**Project:** FinMod — Finance & Accounting LLM

**Current phase:** Controlled fine-tuning and evaluation

**Base model:** SmolLM2-360M

**Current best adapter:**

```text
C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4
```

**Fine-tuning method:** LoRA / PEFT

**Training hardware:** Local Windows PC

**Training device:** CPU

**Current benchmark:**

```text
Benchmark v1: 14/25
Base model:    12/25
Improvement:   +2 questions
```

The current adapter is the **best measured training-only adapter so far**. It should not be interpreted as a production-quality finance model.

---

# 1. Project Goal

The goal of FinMod is to learn and document the complete process of adapting a small open-weight language model toward finance and accounting tasks.

This is an experimental and research-oriented project.

We are **not training an LLM from scratch**.

Instead, the project starts with an existing small language model and investigates whether targeted parameter-efficient fine-tuning can improve capabilities such as:

- Finance terminology
- Accounting concepts
- Financial statement understanding
- Financial numerical reasoning
- Accounting calculations
- Table and text reasoning
- Financial analysis
- Accounting transaction reasoning
- Eventually, accounting workflows and tool use

The emphasis is on **measuring the effect of each intervention**, rather than assuming that more training automatically produces a better model.

---

# 2. Why Start Small?

A major objective of the project is to understand the mechanics of model adaptation.

Using a relatively small model makes it possible to experiment locally and observe the relationship between:

- training data,
- LoRA configuration,
- training loss,
- model behavior,
- evaluation accuracy,
- hardware requirements,
- and inference behavior.

This creates an opportunity to learn the complete workflow without immediately depending on large-scale GPU infrastructure.

The project can later scale to larger models and faster infrastructure once the experimental methodology is understood.

---

# 3. Core Philosophy

## Baseline first

Before claiming that training improved a capability, measure what the base model can already do.

A trained model is only useful as a comparison if there is a meaningful baseline.

---

## Controlled experiments

Where practical, change one major variable at a time.

Examples include:

- LoRA target modules
- LoRA rank
- Loss objective
- Training dataset
- Training examples
- Data composition

This makes it easier to determine **why** a model changed.

---

## Evaluation must remain separate

Evaluation data should not be used as training data.

If an evaluation question appears in the training dataset, its result cannot be treated as evidence of generalization.

This became a particularly important lesson during the project.

---

## Quality over dataset size

The project deliberately avoids the assumption that a large generic finance dataset is automatically better.

A smaller dataset with:

- controlled examples,
- explicit ground truth,
- known difficulty,
- well-defined topics,
- and verifiable answers

can be more useful for experimentation than blindly consuming hundreds of thousands of examples.

---

## Document everything

Significant experiments should record:

- Experiment ID
- Base model
- Dataset version
- Dataset composition
- LoRA configuration
- Training objective
- Hyperparameters
- Hardware
- Training time where available
- Training loss
- Evaluation results
- Observations
- Limitations
- Conclusions

The experiment log is intended to become the permanent record rather than reconstructing experiments from conversation history.

---

# 4. Initial Architecture

The long-term architecture is intentionally broader than fine-tuning alone.

```text
                  Base LLM
                     |
                     v
          Finance / Accounting
             Fine-Tuning
                     |
                     v
                Evaluation
                     |
        +------------+------------+
        |            |            |
        v            v            v
 Financial      Accounting    Numerical
 reasoning      knowledge     accuracy
        |            |            |
        +------------+------------+
                     |
                     v
              RAG + Calculator
                 + Tools
                     |
                     v
          Finance / Accounting AI
```

The intended system should **not** depend on the language model performing every calculation or retrieving every fact internally.

A more robust architecture is:

```text
User
 |
 v
Finance / Accounting Model
 |
 +----> Knowledge / RAG
 |
 +----> Calculator
 |
 +----> Financial data tools
 |
 v
Verified response
```

This distinction is important.

Fine-tuning can improve domain behavior and knowledge, but deterministic calculations, current financial data, and document retrieval are different problems.

---

# 5. Model Strategy

## Original Model v0 Plan

The original project design selected **Qwen3-0.6B** as Model v0 because it was:

- very small,
- suitable for CPU-first experimentation,
- Apache 2.0 licensed,
- supported by the Transformers ecosystem,
- and small enough to make experimentation practical.

The original roadmap also identified several alternative models:

- Llama 3.2 1B Instruct
- Gemma 3 1B IT
- SmolLM2-360M-Instruct

The original project stated that all models would not be fine-tuned initially.

---

## Actual Experimental Model

The experiments subsequently moved to:

```text
SmolLM2-360M
```

This became the actual base model used for the local LoRA experiments.

This distinction is intentionally documented rather than silently rewriting the project's history.

The project therefore demonstrates an important part of the research process itself:

> the planned model and the experimentally used model do not necessarily have to be the same.

---

# 6. Fine-Tuning Approach

FinMod uses **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**.

The base model remains frozen while a relatively small number of trainable parameters are introduced through LoRA adapters.

One of the experiments used:

```text
LoRA rank:        r=16
Target modules:  q_proj
                  k_proj
                  v_proj
                  o_proj
Objective:        response-only loss
```

The r16 configuration contains approximately:

```text
1,638,400 trainable parameters
```

This is only a small fraction of the underlying model parameters.

The project therefore provides a practical example of how domain adaptation can be investigated without updating the complete base model.

---

# 7. Training Data Evolution

The project began with a very small manually controlled finance dataset and progressively expanded it.

The initial training dataset contained **29 examples**.

Later versions added targeted examples covering numerical and accounting reasoning.

The progression was:

```text
finance_tiny.jsonl
        |
        v
finance_tiny_v2.jsonl
        |
        v
finance_tiny_v3.jsonl
        |
        v
finance_tiny_v4.jsonl
```

The final training set used in Exp6 and Exp7 contained:

```text
29 original
+18 v2
+ 8 v3
+10 v4
=65 source examples

-1 excluded exact evaluation example
=64 final training examples
```

---

# 8. Original Dataset Candidates

The original dataset design considered several external sources.

## FinQA

FinQA was selected for financial reasoning involving:

- Financial reports
- Tables
- Numerical reasoning
- Supporting evidence
- Reasoning programs
- Financial question answering

The original plan was to preserve richer information during preprocessing rather than immediately flattening the dataset.

---

## TAT-QA

TAT-QA was selected to complement FinQA with:

- Table + text reasoning
- Financial questions
- Numerical operations
- Multi-step reasoning

---

## NCERT Accounting

The original project also considered accounting material covering:

### Class 11

Approximately 4,408 examples were identified in the original project planning.

Focus areas included:

- Accounting fundamentals
- Accounting process
- Double-entry bookkeeping
- Financial statements
- Accounting principles
- Transactions
- Working capital
- Basic accounting concepts

### Class 12

Approximately 5,730 examples were identified.

Focus areas included:

- Advanced accounting
- Partnership accounting
- Goodwill
- Reconstitution
- Profit-sharing ratios
- Numerical/application questions

Raw external datasets are kept outside the repository where appropriate rather than automatically redistributing them.

---

# 9. Synthetic Accounting Data

An important direction of the project is generating controlled accounting examples programmatically.

Synthetic examples have several advantages for experimentation:

- Exact ground-truth answers
- Automatic verification
- No ambiguity
- Controlled difficulty
- Controlled topic distribution
- Easy generation of variants

Potential areas include:

- Revenue
- COGS
- Gross profit
- Operating expenses
- EBITDA
- EBIT
- PBT
- PAT
- Margins
- ROE
- Debt/equity
- Working capital
- Current ratio
- Quick ratio
- Depreciation
- Interest
- Tax
- Journal entries
- Accruals
- Reconciliation

---

# 10. Actual Training Dataset

The first experimental dataset was intentionally much smaller than the original external dataset candidates.

## `finance_tiny.jsonl`

The initial 29-example dataset covered topics such as:

- Revenue
- Assets
- Liabilities
- Equity
- Accounting equation
- Income statement
- Balance sheet
- Cash flow
- Gross profit
- Operating profit
- Net profit
- Accounts receivable
- Accounts payable
- Working capital
- Simple profit calculations
- Margin calculations
- Depreciation
- Cash flow

---

# 11. Dataset Versioning

## `finance_tiny_v2.jsonl`

The second dataset version introduced targeted formula and finance-reasoning examples.

Examples included percentage increases such as:

```text
$1M -> $1.1M = 10%
$500K -> $600K = 20%
$4M -> $5M = 25%
```

This was intended to test whether explicit examples could improve formula selection and numerical behavior.

---

## `finance_tiny_v3.jsonl`

The third version added examples focused on:

- Explicit percentage calculations
- Accounting equation transactions
- Bank loans
- Owner investment
- Loan repayment
- Equipment purchases
- Cash revenue

---

## `finance_tiny_v4.jsonl`

The fourth version added highly explicit:

- Arithmetic examples
- Accounting transaction examples
- Numerical reasoning examples

This produced the 64-example final training set used by Exp6 and Exp7.

---

# 12. Evaluation Strategy

Evaluation is treated as a separate part of the project.

The original evaluation roadmap included:

- FinQA test
- TAT-QA test
- Professional Accounting MMLU
- FinanceBench
- APEX-Accounting
- A custom accounting benchmark

The project subsequently introduced a smaller fixed benchmark specifically designed for controlled local experimentation.

---

# 13. Early Seven-Question Evaluation

The first evaluation compared the base model and LoRA adapters on the same seven finance/accounting questions.

The questions covered:

1. Gross profit vs net profit
2. Working capital
3. Percentage increase
4. Accounts receivable and operating cash flow
5. Accounting equation impact of a bank loan
6. Gross margin
7. Depreciation under the indirect cash-flow method

The evaluation was deterministic and manually judged.

The evaluation script was later improved to:

- decode only newly generated tokens,
- limit generation length,
- and remove leaked prompt markers such as `### Question:`.

---

# 14. Fixed Benchmark v1

A more robust fixed benchmark was subsequently created.

```text
C:\FinMod\data\finance_eval_v1.jsonl
```

It contains:

| Category | Questions |
|---|---:|
| Finance Concepts | 5 |
| Finance Formulas | 5 |
| Accounting Transactions | 5 |
| Arithmetic | 5 |
| Multi-Step Finance Reasoning | 5 |
| **Total** | **25** |

The benchmark is intended to remain stable across:

- Base model evaluation
- LoRA adapter evaluation
- Future model configurations
- Future tool-assisted architectures

It should **not** be overwritten or trained on.

---

# 15. Experiment Results

The project has now moved beyond planning into measurable controlled experiments.

## Experiment Summary

| Experiment | Main Change | Examples | Final Loss | Result |
|---|---|---:|---:|---|
| Exp1 | LoRA r=8 Q/V, full-sequence loss | 29 | 0.9442 | Initial adaptation |
| Exp2 | Response-only loss | 29 | 0.7641 | Cleaner objective |
| Exp3 | Q/K/V/O targets | 29 | 0.6072 | Better training fit |
| Exp4 | Added targeted v2 data | 46 | 0.5550 | Some formula improvements |
| Exp5 | Added v3 reasoning data | 54 | 0.5092 | Better training fit |
| Exp6 | Added v4 arithmetic/accounting data | 64 | 0.5142 | Some accounting improvements |
| Exp7 | LoRA rank increased to 16 | 64 | 0.4370 | Lowest training loss |
| Exp8 | Fixed Benchmark v1 on best LoRA | — | — | **14/25** |
| Exp9 | Fixed Benchmark v1 on base model | — | — | **12/25** |

---

# 16. Exp1 — First LoRA Training Run

The first experiment tested whether a small LoRA adapter could successfully train on the local CPU environment.

Configuration:

```text
Base model:       SmolLM2-360M
LoRA rank:        r=8
LoRA alpha:       16
Target modules:  q_proj, v_proj
Dropout:          0.05
Bias:             none
Task:             CAUSAL_LM
Device:           CPU
Max length:       128
Batch size:       1
Epochs:           3
Learning rate:    2e-4
Trainable params: 819,200
```

Training loss:

```text
Epoch 1    1.8550
Epoch 2    1.2769
Epoch 3    0.9442
```

The model learned portions of the tiny training set, but several important failures remained.

For example:

- Working-capital calculation improved.
- Some finance definitions improved.
- Percentage-increase reasoning remained wrong.
- Gross-margin calculation regressed.
- Some causal financial explanations remained incorrect.

The experiment established an important principle:

> A falling training loss does not prove generalization.

---

# 17. Exp2 — Response-Only Loss

Exp2 kept the LoRA configuration and dataset fixed while changing the training objective.

Instead of training on the entire formatted sequence, the loss was calculated only on response tokens.

Configuration:

```text
LoRA rank:        r=8
Target modules:  q_proj, v_proj
Epochs:           3
Learning rate:    2e-4
Max length:       128
Trainable params: 819,200
```

Training loss:

```text
Epoch 1    1.2122
Epoch 2    0.9576
Epoch 3    0.7641
```

This produced cleaner response behavior, but did not solve the fundamental numerical and reasoning failures.

The experiment demonstrated that:

> A better instruction-tuning objective does not automatically produce better reasoning.

---

# 18. Exp3 — Q/K/V/O LoRA Targets

Exp3 tested whether Q/V-only adaptation was too restrictive.

The target modules were expanded to:

```text
q_proj
k_proj
v_proj
o_proj
```

while retaining response-only loss and r=8.

Training loss:

```text
Epoch 1    1.1671
Epoch 2    0.8516
Epoch 3    0.6072
```

Trainable parameters increased to:

```text
1,638,400
```

Some conceptual behavior improved, but repeated arithmetic and formula-selection failures remained.

This suggested that the problem was not simply insufficient LoRA target coverage.

---

# 19. Exp4 — Targeted v2 Data

Exp4 added targeted formula and finance-reasoning examples.

Dataset:

```text
29 original examples
+18 targeted examples
=47 source examples

1 evaluation example excluded
=46 training examples
```

Training loss:

```text
Epoch 1    1.1009
Epoch 2    0.7757
Epoch 3    0.5550
```

The experiment improved some targeted behaviors, including gross-margin handling, but arithmetic remained unreliable.

This was an early demonstration that **adding examples aimed directly at a failure mode can change model behavior without necessarily fixing the underlying capability**.

---

# 20. Exp5 — Numerical and Accounting Reasoning

Exp5 introduced v3 examples focused on numerical and accounting reasoning.

Final training dataset:

```text
54 examples
```

Final training loss:

```text
0.5092
```

Training fit continued to improve, but later evaluation showed that some important numerical reasoning failures remained.

This reinforced the distinction between:

```text
training-set fit
```

and

```text
generalization
```

---

# 21. Exp6 — Explicit Arithmetic and Accounting Examples

Exp6 added v4 examples designed around explicit arithmetic and accounting reasoning.

Final training set:

```text
64 examples
```

Training loss:

```text
0.5142
```

One notable improvement was the bank-loan accounting case.

However, the percentage-increase failure remained.

Some regressions also appeared.

This showed that adding examples can improve one behavior while leaving another unchanged or potentially affecting other behaviors.

---

# 22. Exp7 — Increasing LoRA Rank

Exp7 kept the v4 dataset but increased LoRA rank:

```text
r=8  ->  r=16
```

Target modules remained:

```text
q_proj
k_proj
v_proj
o_proj
```

Final training loss:

```text
0.4370
```

This was the lowest training loss achieved in the sequence of experiments.

However, the same seven-question evaluation still failed the percentage-increase question.

This was a useful demonstration that:

> Increasing adaptation capacity can improve training fit without necessarily solving a capability problem.

---

# 23. Diagnostic A — Arithmetic and Memorization

A dedicated arithmetic diagnostic was created to investigate whether the repeated numerical failures were caused by:

- lack of arithmetic capability,
- insufficient examples,
- memorization,
- or some combination of these.

The diagnostic showed that arithmetic/composition remained unreliable.

It also uncovered an important evaluation contamination problem involving one of the percentage-increase questions.

---

# 24. Diagnostic B — Structured Reasoning

A second diagnostic tested whether explicitly asking for step-by-step reasoning could fix the observed failures.

The diagnostic helped identify weaknesses in:

- composition,
- multi-step reasoning,
- and causal consistency.

The result was not simply that "chain-of-thought prompting fixes the model."

Instead, the diagnostic reinforced the need to understand which capabilities are actually missing.

---

# 25. Capability Evaluation

A broader capability evaluation was also performed.

The r16 adapter showed its strongest behavior on:

- formulas,
- arithmetic,

while:

- accounting,
- multi-step reasoning

remained mixed.

Manual scoring inconsistency was also identified as a limitation of the evaluation process.

This is another reason the project is moving toward more controlled and automatically verifiable evaluation wherever practical.

---

# 26. Benchmark v1 Results

The fixed 25-question benchmark produced the first stable comparison between the base model and the current best LoRA adapter.

### Base Model

```text
12 / 25
```

### Current Best LoRA

```text
14 / 25
```

### Difference

```text
+2 questions
```

The important conclusion is not that the model suddenly became a capable finance model.

The more defensible conclusion is:

> The current LoRA configuration produced a measurable improvement over the base model on the fixed benchmark, but the improvement is modest and substantial weaknesses remain.

---

# 27. Important Contamination Warning

One of the project's most important lessons came from the question:

```text
Revenue increased from $2 million to $2.4 million.
What is the percentage increase?
```

This exact question appeared in the v4 training data.

Therefore, evaluation results on that exact question **cannot be considered valid evidence of generalization**.

More interestingly, the model still answered `10%` in several runs despite having been exposed to the example.

This demonstrated two separate problems:

1. Evaluation contamination can invalidate an apparent result.
2. Exposure to the correct example does not guarantee reliable arithmetic behavior.

The fixed Benchmark v1 was subsequently created with different wording/numbers where possible and must remain separate from training data.

---

# 28. What the Experiments Have Taught So Far

Several conclusions have emerged from the experiments.

## 1. LoRA works on the local machine

The project successfully trained LoRA adapters on CPU.

The base model weights remain frozen while the adapter parameters are trained.

---

## 2. Training loss is not enough

Training loss decreased substantially across experiments.

But lower training loss did not consistently translate into correct answers on held-out questions.

---

## 3. Response-only loss helps the training objective

Response-only loss produced cleaner instruction-tuning behavior.

But it did not automatically solve reasoning.

---

## 4. More LoRA capacity is not automatically better

Moving from r=8 to r=16 reduced training loss.

It did not eliminate the recurring numerical failure.

---

## 5. More examples are not automatically better

Targeted examples improved some behaviors but also produced unchanged failures and occasional regressions.

Dataset composition matters.

---

## 6. Numerical reasoning needs special treatment

A language model can reproduce the form of a financial calculation while still performing the arithmetic incorrectly.

This suggests that reliable finance systems should not assume that fine-tuning alone is sufficient for numerical correctness.

---

## 7. Evaluation design matters as much as training

A contaminated question can produce misleading conclusions.

A stable held-out benchmark is therefore essential.

---

# 29. Environment

The experiments were performed in a Windows local environment.

```text
OS / shell:
Windows PowerShell

Project root:
C:\FinMod

Python environment:
C:\FinMod\.venv

Base model:
C:\FinMod\models\SmolLM2-360M

Training / evaluation device:
CPU

CUDA:
torch.cuda.is_available() == False

GPU:
NVIDIA GT 710 present,
but not used for PyTorch acceleration
```

Observed training resource usage was approximately:

```text
RAM:
12.6–12.9 GB of 15.8 GB

CPU:
approximately 52–77%

CPU frequency:
approximately 3.68–3.78 GHz
```

Generation during evaluations was deterministic:

```python
do_sample=False
```

The project therefore demonstrates that small-scale PEFT experimentation is possible on relatively modest local hardware, although training is naturally much slower than on a suitable GPU.

---

# 30. Repository Structure

The project has evolved from the original planned structure into an experiment-oriented repository.

The documented working structure includes:

```text
C:\FinMod\
│
├── .venv\
│
├── models\
│   └── SmolLM2-360M\
│
├── data\
│   ├── finance_tiny.jsonl
│   ├── finance_tiny_v2.jsonl
│   ├── finance_tiny_v3.jsonl
│   ├── finance_tiny_v4.jsonl
│   └── finance_eval_v1.jsonl
│
├── scripts\
│   ├── 01_attach_lora.py
│   ├── 02_train_lora.py
│   ├── 03_evaluate_lora.py
│   ├── 04_arithmetic_diagnostic.py
│   ├── 06_capability_eval.py
│   ├── 07_score_capability.py
│   ├── 08_run_eval_v1.py
│   └── 09_run_base_eval_v1.py
│
├── adapters\
│   ├── finance_lora_r8_qv\
│   ├── finance_lora_r8_qv_response_only\
│   ├── finance_lora_r8_qkvo_response_only\
│   ├── finance_lora_r8_qkvo_response_only_v2\
│   ├── finance_lora_r8_qkvo_response_only_v3\
│   ├── finance_lora_r8_qkvo_response_only_v4\
│   └── finance_lora_r16_qkvo_response_only_v4\
│
└── outputs\
    ├── finance_eval_v1_r16_qkvo_v4_results.jsonl
    └── finance_eval_v1_base_results.jsonl
```

Large model weights, local virtual environments, runtime artifacts, and external datasets are intentionally not treated as normal source-controlled project files.

---

# 31. External Data and Reproducibility

External datasets should be obtained from their respective official or original sources rather than unnecessarily redistributing them inside this repository.

The repository should therefore contain:

- dataset documentation,
- provenance,
- preprocessing code,
- dataset versions or identifiers,
- and instructions for obtaining external data,

rather than assuming that raw third-party datasets belong in the Git repository.

This also keeps the repository manageable and makes the distinction between:

```text
project code
```

and

```text
external source data
```

clear.

---

# 32. Evaluation Principles

The project follows several evaluation principles.

### Never evaluate only training loss

Training loss measures optimization against the training objective.

It does not measure generalization.

### Keep a fixed benchmark

Benchmark v1 provides a stable reference point.

### Use a base-model control

A fine-tuned model should be compared against the corresponding base model.

### Test individual capabilities

Finance knowledge, formulas, arithmetic, accounting transactions, and multi-step reasoning should not automatically be collapsed into one number.

### Prefer automatic verification where possible

Synthetic numerical/accounting tasks are particularly useful because their answers can often be independently calculated.

### Track failures, not only successes

A wrong answer can reveal more about the model's limitations than a correct memorized answer.

---

# 33. Future Work

The next stages of the project are expected to investigate several directions.

## Better held-out evaluation

Expand the controlled benchmark while maintaining strict separation from training data.

---

## Better numerical evaluation

Build more automatically verifiable numerical tasks covering:

- percentages,
- margins,
- ratios,
- working capital,
- financial statement calculations,
- accounting equation changes,
- multi-step calculations.

---

## Better accounting reasoning

Expand controlled accounting transaction scenarios.

Examples:

- owner investment,
- borrowing,
- repayment,
- asset purchases,
- revenue recognition,
- accruals,
- depreciation,
- expense recognition,
- working-capital movements.

---

## Dataset experiments

Test whether carefully designed mixtures of:

- conceptual finance,
- accounting knowledge,
- numerical examples,
- reasoning examples,

produce better generalization than simply increasing dataset size.

---

## Larger models

Once the experimental methodology is sufficiently mature, compare the behavior of larger small-scale models.

The purpose would not be to abandon the local experimental approach, but to understand how model capacity changes the results.

---

## Tool-assisted architecture

Investigate a system where the model delegates tasks to deterministic components.

For example:

```text
User question
     |
     v
Finance Model
     |
     +----> RAG / document retrieval
     |
     +----> Calculator
     |
     +----> Financial data source
     |
     v
Verified answer
```

This is particularly important for financial applications where numerical correctness and current information matter.

---

# 34. Long-Term Direction

The long-term goal is not simply a chatbot that knows accounting definitions.

The intended direction is a finance/accounting AI combining:

- A domain-adapted language model
- Financial reasoning
- Accounting knowledge
- Financial document understanding
- Retrieval-Augmented Generation
- Deterministic calculations
- External financial data
- Accounting workflow tools

The model is only one component of that system.

The broader objective is to understand where each component provides value and where relying on the language model alone becomes unreliable.

---

# 35. Deployment Strategy

Ollama is **not the core training framework**.

Its intended role in the project is:

- Local inference
- Quick model testing
- Later deployment experimentation

Training is performed using the Transformers and PEFT/LoRA ecosystem.

After training, models may eventually be converted or quantized for deployment formats such as GGUF/Ollama where appropriate.

A separate online AI platform available through a friend may be considered later for:

- larger experiments,
- faster training/inference,
- custom OpenAI-compatible endpoint testing,
- and product-oriented experiments.

The current philosophy remains:

> **Local research first. Scale later.**

---

# 36. Current Best Model

The current best measured adapter is:

```text
finance_lora_r16_qkvo_response_only_v4
```

It achieved:

```text
14 / 25
```

on the fixed Benchmark v1.

The corresponding base model achieved:

```text
12 / 25
```

Therefore:

```text
LoRA improvement = +2 / 25 questions
```

This is a useful experimental result, but it is **not evidence that the model is ready for production finance use**.

The current adapter should be described as:

> **the current best training-only adapter in this experiment series**

rather than as a production finance model.

---

# 37. Project Status Summary

The project has progressed through several stages:

```text
Initial architecture
        |
        v
Model selection
        |
        v
Local environment
        |
        v
Tiny controlled dataset
        |
        v
First LoRA experiment
        |
        v
Loss-objective experiment
        |
        v
LoRA target-module experiment
        |
        v
Targeted dataset iterations
        |
        v
Arithmetic / reasoning diagnostics
        |
        v
Fixed benchmark
        |
        v
Base vs trained comparison
        |
        v
Current controlled research phase
```

The project is therefore no longer merely a model-selection or dataset-design exercise.

It has reached the stage where **controlled experiments and measured evaluation results drive the next decisions**.

---

# 38. Lessons for the Next Phase

The central lesson so far is simple:

> **Making a language model train harder is not the same thing as making it reason reliably.**

The experiments have demonstrated that:

```text
Lower training loss
        ≠
Better generalization
```

and:

```text
More training examples
        ≠
Better reasoning
```

and:

```text
More LoRA capacity
        ≠
Reliable arithmetic
```

A useful finance model will likely require a combination of:

```text
Domain adaptation
        +
Carefully designed data
        +
Strong evaluation
        +
Deterministic computation
        +
Retrieval
        +
External financial data
        +
Workflow tools
```

That is the direction FinMod will investigate next.

---

# 39. Project Identity

**Project name:** FinMod

**Repository:** `training-ai-model`

**Primary objective:** Learn and document the end-to-end process of adapting small open-weight language models for finance and accounting.

**Current base model:** SmolLM2-360M

**Fine-tuning method:** LoRA / PEFT

**Current best adapter:** `finance_lora_r16_qkvo_response_only_v4`

**Current fixed benchmark:** 25 questions

**Current best result:** 14/25

**Base-model result:** 12/25

**Current status:** Controlled fine-tuning and evaluation research

---

## Final note

This README intentionally **does not pretend the project has achieved more than it has**.

The interesting part of FinMod at this stage is not “we trained a finance LLM.”

It is that the project now has a traceable experimental progression:

```text
baseline
   ↓
LoRA
   ↓
response-only objective
   ↓
broader target modules
   ↓
targeted datasets
   ↓
higher LoRA rank
   ↓
diagnostics
   ↓
held-out benchmark
   ↓
base vs trained comparison
```

That makes the repository much more valuable as a **research/engineering project and résumé artifact**, because someone looking at it can see not just a model file, but the process used to determine what actually worked.
