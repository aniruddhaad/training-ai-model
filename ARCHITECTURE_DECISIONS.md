# FinMod — Architecture Decisions

## Document Status

**Version:** 0.1

**Project:** FinMod — Finance & Accounting LLM

**Current decision phase:** Model v0 / Dataset v0.1 / Evaluation v0

---

# 1. Decision Summary

| Component | Decision |
|---|---|
| Base model | Qwen3-0.6B |
| Training method | LoRA / PEFT |
| Initial hardware strategy | CPU-first |
| Primary financial reasoning dataset | FinQA |
| Table/text reasoning dataset | TAT-QA |
| Accounting foundation | NCERT Accounting 11 |
| Advanced accounting | NCERT Accounting 12 |
| Verified arithmetic | Our own synthetic accounting data |
| Accounting knowledge evaluation | Professional Accounting MMLU |
| Financial QA evaluation | FinanceBench |
| Accounting workflow evaluation | APEX-Accounting |
| Local inference/deployment | Ollama later |
| Online platform | Later experimentation |
| Training from scratch | Rejected |

---

# 2. Model Decision

## Selected: Qwen3-0.6B

### Why

Qwen3-0.6B is currently the best fit for the first controlled experiment.

Reasons:

1. Very small model size.
2. Practical for experimentation on our available hardware.
3. Official Transformers support.
4. Apache 2.0 licensing is attractive for research and future flexibility.
5. Small enough that we can investigate the complete fine-tuning workflow ourselves.
6. Large enough to be a meaningful language-model experiment rather than an extremely tiny toy model.
7. Qwen3 gives us a modern architecture that can later be compared with other model families.

The key research question is not whether Qwen is universally the best model.

The question is:

> How much finance/accounting capability can we add to a small general-purpose model through carefully targeted training?

---

# 3. Hardware Decision

## CPU-first

The current local machine has:

- Intel i7 CPU
- 16 GB RAM
- NVIDIA GT 710
- 2 GB GPU memory

The GT 710 is not considered a practical GPU for modern LLM fine-tuning.

Therefore the first experiments should be designed around CPU execution and small models.

This constraint is intentional.

It forces us to understand:

- Dataset preparation
- Memory usage
- LoRA
- Training configuration
- Evaluation
- Quantization
- Inference

without hiding the process behind a large cloud GPU.

If a later experiment genuinely needs more compute, the friend's online platform can be used.

---

# 4. Why Not Train From Scratch?

Rejected.

Training a useful modern LLM from scratch would require:

- Huge datasets
- Large compute resources
- Extensive infrastructure
- Long training time

It would also distract from the actual research goal.

The project is about domain adaptation.

Therefore:

    Existing open-weight model
            |
            v
       Domain data
            |
            v
       LoRA / PEFT
            |
            v
       Finance model

---

# 5. Why LoRA / PEFT?

Selected for the first training approach.

Advantages:

- Updates only a small number of parameters.
- Lower memory requirements than full fine-tuning.
- Easier experimentation.
- Smaller checkpoints.
- Well suited to controlled experiments.
- Makes it easier to compare multiple datasets/models.

Full fine-tuning can be investigated later if hardware permits.

---

# 6. Why FinQA?

## Decision: Core training dataset

FinQA is an official IBM Research dataset focused on numerical reasoning over financial reports.

It contains approximately:

- 2.8k financial reports
- 8k QA pairs

Important characteristics:

- Financial report context
- Tables
- Questions
- Answers
- Supporting evidence
- Reasoning programs

The reasoning-program information is particularly valuable.

We should not immediately flatten FinQA into question/answer pairs and throw away its structure.

Potential training representations can be compared.

For example:

    Context
       +
    Question
       +
    Evidence
       +
    Reasoning
       =
    Answer

versus:

    Context
       +
    Question
       =
    Answer

This creates a future research experiment around reasoning supervision.

---

# 7. Why TAT-QA?

## Decision: Core training/evaluation candidate

TAT-QA complements FinQA.

It focuses on financial question answering involving:

- Tables
- Narrative text
- Numerical reasoning
- Multi-step operations

FinQA and TAT-QA therefore cover overlapping but not identical reasoning behavior.

We want the model to learn financial reasoning from both structured and unstructured information.

The exact TAT-QA version, split, and license will be recorded during Dataset v0.1 finalization.

---

# 8. Why NCERT Accounting Class 11?

## Decision: Core accounting training candidate

The Class 11 dataset contains approximately 4,408 examples.

It focuses primarily on accounting foundations.

Examples include:

- Accounting fundamentals
- Accounting process
- Double-entry bookkeeping
- Financial statements
- Accounting principles
- Transactions
- Working capital
- Basic accounting concepts

This fills an important gap in FinQA.

FinQA teaches the model to reason about financial reports.

NCERT Accounting teaches the model what accounting concepts mean.

---

# 9. Why NCERT Accounting Class 12?

## Decision: Core accounting training candidate

The Class 12 dataset contains approximately 5,730 examples.

It introduces more advanced accounting concepts, including:

- Partnership accounting
- Goodwill
- Reconstitution
- Profit-sharing ratios
- Admission/retirement concepts
- Numerical and application-oriented questions

Class 11 and Class 12 therefore create a natural progression:

    Class 11
    Accounting foundations
          |
          v
    Class 12
    Advanced accounting
          |
          v
    Financial reasoning
    and analysis

The datasets show MIT at the Hugging Face repository level, but underlying-content provenance should be documented before any commercial redistribution.

---

# 10. Why Synthetic Accounting Data?

## Decision: Create our own

Synthetic data will provide something the external datasets cannot guarantee:

**exactly verifiable answers.**

We can generate accounting scenarios with Python.

For example:

    Revenue = 10,000
    COGS = 6,000
    Operating expenses = 2,000

    Gross Profit = 4,000
    EBIT = 2,000

Because we control the underlying numbers, the expected result can be automatically verified.

This is valuable for:

- Arithmetic
- Ratios
- Financial statements
- Journal entries
- Reconciliation
- Accounting calculations
- Controlled difficulty

It also gives us a clean benchmark for determining whether the model actually learned to calculate.

---

# 11. Why Not Use a Huge Finance Instruction Dataset?

Deferred.

Datasets with 177k or 500k+ finance examples are tempting.

However, using a huge heterogeneous dataset immediately creates problems:

- Harder to understand what improved the model.
- More duplication.
- Potentially mixed provenance.
- More training time.
- More CPU cost.
- Less controlled experimentation.

The first experiment should answer a scientific question.

Therefore:

> Start with a smaller, carefully composed dataset.

Large datasets can be added later if evaluation demonstrates a specific capability gap.

---

# 12. Accounting Dataset Decisions

## `KadamParth/NCERT_Accounting_11th`

Decision: **Keep as training candidate**

Reason:

- ~4,408 examples
- Accounting-focused
- Instruction/QA format
- Rich metadata
- Useful fundamentals

Caveat:

- Repository shows MIT.
- Underlying NCERT provenance should be documented before commercial redistribution.

---

## `KadamParth/NCERT_Accounting_12th`

Decision: **Keep as training candidate**

Reason:

- ~5,730 examples
- More advanced accounting
- Natural complement to Class 11
- Instruction/QA format
- Rich metadata

Same provenance/licensing caveat as Class 11.

---

## `algohype/accounting`

Decision: **Do not make core Dataset v0.1**

Reason:

- Only ~315 examples.
- Useful accounting concepts.
- However, the repository's content includes apparent third-party AccountingCoach material and provenance/licensing requires additional investigation.

The content may be useful later, but the NCERT datasets provide a larger and more structured accounting component.

---

## `Lots-of-LoRAs/task728_mmmlu_answer_generation_professional_accounting`

Decision: **Evaluation/reference, not core training**

Reason:

- Small training split.
- Primarily a generated MMLU task formulation.
- Teaches multiple-choice answer behavior.
- Not ideal for teaching explanatory accounting behavior.

It remains useful for understanding task-format performance.

---

## `jacobwelsh/accounting`

Decision: **Reference/RAG candidate, not current training data**

Reason:

- Contains accounting-standard PDFs.
- Useful as a knowledge source.
- Dataset documentation and underlying redistribution rights are unclear.

Potential future use:

    User question
        |
        v
    Retrieve accounting standard
        |
        v
    Finance model
        |
        v
    Grounded answer

---

# 13. Evaluation Architecture

Evaluation is deliberately separated from training.

## Accounting knowledge

Professional Accounting MMLU.

Purpose:

> Does the model know accounting concepts?

---

## Financial numerical reasoning

FinQA test.

Purpose:

> Can the model reason over financial reports and numbers?

---

## Table + text reasoning

TAT-QA test.

Purpose:

> Can the model combine narrative financial information with tables?

---

## Financial QA

FinanceBench.

Purpose:

> Can the model answer realistic financial questions using evidence?

---

## Accounting workflow

APEX-Accounting.

Purpose:

> Can the model perform more realistic accounting work?

Examples include:

- Reconciliation
- Data entry
- Variance analysis
- Schedules
- Accruals

APEX must remain evaluation-only.

---

# 14. Why Multiple Evaluation Sets?

A single benchmark is insufficient.

A model could become better at accounting terminology without becoming better at numerical reasoning.

Another model could improve arithmetic but still fail actual accounting workflows.

Therefore:

    Accounting Knowledge
            |
            +----> MMLU Professional Accounting
            |
    Financial Reasoning
            |
            +----> FinQA / TAT-QA
            |
    Financial QA
            |
            +----> FinanceBench
            |
    Accounting Workflow
            |
            +----> APEX
            |
    Verified Calculation
            |
            +----> Our synthetic benchmark

---

# 15. Baseline Decision

Before any fine-tuning:

1. Run Qwen3-0.6B on the evaluation suite.
2. Record results.
3. Record inference time.
4. Record RAM usage.
5. Save representative answers.
6. Establish baseline errors.

Only then begin fine-tuning.

Without a baseline, we cannot confidently say the training helped.

---

# 16. Planned Experiment Sequence

## Experiment 0

    Qwen3-0.6B

No finance fine-tuning.

Purpose: baseline.

---

## Experiment 1

    Qwen3-0.6B
          +
    FinQA/TAT-QA

Purpose: measure financial reasoning improvement.

---

## Experiment 2

    Qwen3-0.6B
          +
    FinQA/TAT-QA
          +
    NCERT Accounting

Purpose: measure accounting knowledge improvement.

---

## Experiment 3

    Qwen3-0.6B
          +
    Previous data
          +
    Synthetic verified accounting

Purpose: measure numerical/accounting calculation improvement.

---

## Experiment 4

Combined best-performing configuration.

Purpose: determine whether the components complement each other.

---

# 17. Future Architecture

Fine-tuning is only the first stage.

The long-term architecture is:

                    User
                      |
                      v
              Finance AI Model
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
         RAG      Calculator   Financial
                               Data/API
          |           |           |
          +-----------+-----------+
                      |
                      v
               Verified response

The model should eventually know when to:

- Answer directly.
- Retrieve accounting knowledge.
- Perform a deterministic calculation.
- Query financial data.
- Explain the result.

---

# 18. Key Research Question

The project should ultimately answer:

> Can a very small general-purpose language model be transformed into a useful finance/accounting reasoning model using a relatively small, carefully selected and verified dataset?

Secondary questions:

1. How much does financial reasoning data help?
2. How much does accounting knowledge data help?
3. Does synthetic verified arithmetic improve numerical accuracy?
4. Does reasoning supervision help a 0.6B model?
5. Does better benchmark performance translate into real accounting workflow performance?
6. How much capability can we obtain without a large GPU?

---

# 19. Current Architecture Freeze

## Model v0

**Qwen3-0.6B**

## Dataset v0.1 direction

**Training:**

- FinQA
- TAT-QA
- NCERT Accounting 11
- NCERT Accounting 12
- Synthetic verified accounting

**Evaluation:**

- FinQA test
- TAT-QA test
- Professional Accounting MMLU
- FinanceBench
- APEX-Accounting
- Our own synthetic accounting benchmark

## Training framework

Transformers + PEFT/LoRA.

## Inference/deployment

Ollama later.

## External compute/platform

Friend's online AI platform later, after local research establishes a baseline.

---

# 20. Immediate Next Decision

Before downloading and training:

1. Finalize exact dataset proportions.
2. Inspect and clean the source datasets.
3. Define the unified training schema.
4. Define train/validation/evaluation boundaries.
5. Define the first benchmark questions.
6. Document provenance/licensing.
7. Freeze Dataset v0.1.
8. Freeze Evaluation v0.
9. Set up the local training environment.

After that, the project moves from research/design into implementation.

---

**FinMod — Architecture Decisions v0.1**
