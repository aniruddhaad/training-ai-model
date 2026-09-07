# SmolLM2-360M Accounting Adaptation — Experiment Results

## Experiment

**Experiment:** SmolLM2-360M Accounting Adaptation

**Purpose:**  
Evaluate whether a small open-weight language model can be adapted to accounting knowledge using parameter-efficient fine-tuning, and establish a reproducible baseline for comparison with larger models.

---

## Base Model

**Model:** `HuggingFaceTB/SmolLM2-360M`

**Approximate parameters:** ~361.8M

**Model type:** Small decoder language model

The base model was downloaded locally and used as the frozen base model for LoRA fine-tuning.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Fine-tuning method | LoRA |
| Target modules | Q/K/V/O projections |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Trainable parameters | 1,638,400 |
| Trainable parameter ratio | 0.4508% |
| Loss | Response-only |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Batch size | 16 |
| Gradient accumulation | 1 |
| Maximum sequence length | 256 |
| Random seed | 42 |

The base model parameters remained frozen. Only the LoRA adapter parameters were trained.

Response-only loss means that prompt tokens were masked with `-100`; the training loss was therefore calculated only on the answer portion of each example.

---

## Dataset

The final cleaned accounting datasets were:

| Dataset | Examples |
|---|---:|
| Class 11 | 4,232 |
| Class 12 | 5,133 |
| Combined | 9,365 |
| Removed by exact Q+A deduplication | 205 |
| Final usable dataset | **9,160** |

The combined dataset was deterministically shuffled using seed `42`.

### Train / Validation Split

A 95% / 5% split was used.

| Split | Examples |
|---|---:|
| Training | **8,702** |
| Validation | **458** |
| Total | **9,160** |

The validation set was held out from training.

---

## Training Hardware

**GPU:** NVIDIA Tesla T4

**VRAM:** 15 GB

Training was performed in Google Colab.

---

## Training Results

**Training time:** ~5.18 minutes

**Final training loss:** 1.2073569

**Final validation loss:** 1.187842

Training loss progression:

| Training point | Loss |
|---|---:|
| Step 100 | 1.257839 |
| Step 200 | 1.223045 |
| Step 300 | 1.204347 |
| Step 400 | 1.193599 |
| Step 500 | 1.188381 |
| Final | **1.187842 validation loss** |

---

## Adapter

The production adapter from this experiment was:

`accountancy_lora_r8_qkvo_v1`

The adapter was saved separately from the base model.

The trained LoRA adapter contains approximately 1.64M trainable parameters rather than modifying the ~361.8M base-model parameters directly.

---

# Evaluation

Several evaluation stages were performed during the experiment.

## 1. Validation Evaluation

A held-out validation set containing 458 examples was used to compare the base model against the fine-tuned model.

The evaluation was performed deterministically using:

- temperature-free generation
- `do_sample=False`
- left padding
- maximum generation length of 128 tokens

The fine-tuned model produced substantially different responses from the base model.

One observable behavioral change was a reduction in continuation of the dataset's `Question:` structure.

However, behavioral change alone does not establish accounting correctness.

---

## 2. Semantic Evaluation

Multiple attempts were made to automatically score accounting correctness.

The first generic semantic scorer was rejected because it could incorrectly treat later generated `Question:` continuations as part of the answer and therefore award credit for numbers that did not belong to the model's actual first answer.

A second evaluator was intentionally more conservative, but produced too many `REVIEW` cases to serve as a reliable automated accuracy measurement.

Therefore, neither automated semantic scorer is treated as the official model accuracy.

---

## 3. Gold Evaluation

A fixed **FinMod Gold Evaluation V1** benchmark was created from the held-out validation questions.

It contains:

- 100 questions
- 50 Class 11 questions
- 50 Class 12 questions
- 57 objective/accounting-mechanism questions
- 43 conceptual questions
- Easy / Medium / Hard difficulty coverage

Manual accounting audits were performed on the benchmark.

The audits showed that the fine-tuned model improved on some accounting concepts and explanations, but still produced substantial errors in numerical reasoning, accounting mechanisms, and interpretation of questions.

The manual audit therefore supports the conclusion that the model learned meaningful accounting patterns, but did not reach reliable accounting reasoning.

Some benchmark expected answers themselves contained questionable accounting treatment, so the benchmark should continue to be refined rather than treated as an immutable ground truth.

---

# Observations

The experiment demonstrated several important findings.

### 1. LoRA works effectively on a small model

Only approximately **0.45% of the model's parameters** were trainable, yet the fine-tuned model's behavior changed substantially.

This demonstrates the usefulness of parameter-efficient fine-tuning for adapting a small open-weight model.

### 2. The accounting dataset was learned

The model demonstrated improved familiarity with accounting terminology, relationships, formulas, and common accounting concepts after fine-tuning.

Examples include:

- accounting equation
- assets, liabilities and equity
- depreciation
- working capital
- accounts receivable/payable
- financing transactions
- accounting classifications

### 3. Learning accounting terminology is not the same as reliable accounting reasoning

The model could produce accounting-looking answers while still making mistakes in:

- numerical calculations
- percentage reasoning
- transaction-state changes
- interpretation of accounting events
- multi-step reasoning

This distinction is important for the next experiment.

### 4. Evaluation methodology matters

Generic text similarity and naive numerical matching were not sufficient for evaluating accounting reasoning.

The experiment therefore motivated the development of a dedicated accounting-aware evaluation methodology.

---

# Final Conclusion

> **Result:** The model clearly learned from the accounting dataset and changed its response behaviour substantially compared with the base model. However, manual evaluation showed that accounting reasoning remained inconsistent. The model is therefore considered a successful fine-tuning/learning experiment but not yet a production-quality accounting model.

The primary goal of this experiment was **learning and documenting the end-to-end adaptation process**, rather than producing a production-ready accounting model.

That goal was successfully achieved.

---

# Baseline for Future Experiments

This experiment establishes the first reproducible FinMod accounting baseline.

Future models should be compared against this experiment using, where practical:

- the same accounting evaluation questions
- the same held-out validation methodology
- the same accounting-aware evaluation principles
- comparable reporting of training configuration
- training and validation loss
- parameter efficiency
- qualitative accounting reasoning
- numerical accuracy
- multi-step reasoning performance

The next major experiment can therefore be evaluated not merely by whether its training loss is lower, but by whether it produces **more reliable accounting reasoning than the SmolLM2-360M baseline**.

---

## Reproducibility

**Base model:** `HuggingFaceTB/SmolLM2-360M`

**Canonical datasets:**

- `data/Accountancy_11th_FINAL.csv`
- `data/Accountancy_12th_FINAL.csv`

**Training script:**

- `scripts/16_train_accountancy_lora_r8_v1.py`

**Evaluation benchmark:**

- `data/FinMod_Gold_Evaluation_V1_100.csv`

**Historical experiment artifacts:**

- `archive/360M_experiment/`

**Production adapter:**

- `accountancy_lora_r8_qkvo_v1`

This experiment is preserved in the repository so that the results can be revisited and compared with future model adaptations.