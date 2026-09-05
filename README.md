# FinMod — Finance & Accounting LLM

## Project Status

**Current phase:** Dataset and architecture design

**Model v0:** Qwen3-0.6B

**Training approach:** Parameter-Efficient Fine-Tuning (LoRA/PEFT)

**Hardware target:** Local Windows PC, CPU-first

---

## 1. Project Goal

The goal of FinMod is to learn and document the complete process of adapting a small open-weight language model into a useful finance and accounting model.

This is an experimental/research project.

We are intentionally NOT training an LLM from scratch.

Instead, we will start with a small existing model and progressively teach it:

- Finance terminology
- Accounting concepts
- Financial statement understanding
- Financial numerical reasoning
- Accounting calculations
- Table + text reasoning
- Financial analysis
- Eventually, accounting workflows and tool use

The emphasis is on understanding the process and measuring whether each intervention actually improves the model.

---

## 2. Core Philosophy

### Baseline first

Before training, measure what the base model can already do.

### Controlled experiments

Change one major variable at a time wherever practical.

### Evaluation must remain separate

Evaluation data must not leak into training.

### Quality over dataset size

We prefer a small, carefully designed dataset over blindly consuming hundreds of thousands of generic finance examples.

### Document everything

Each significant experiment should record:

- Experiment ID
- Base model
- Dataset version
- Dataset composition
- Hyperparameters
- Hardware
- Training time
- Baseline result
- Post-training result
- Observations
- Conclusion

---

## 3. Architecture

Initial architecture:

    Base LLM
        |
        v
    Finance / Accounting Fine-Tuning
        |
        v
    Evaluation
        |
        +--> Financial reasoning
        +--> Accounting knowledge
        +--> Numerical accuracy
        +--> Workflow capability
        |
        v
    RAG + Calculator + Tools
        |
        v
    Finance / Accounting AI

The eventual system should not depend on the LLM performing everything itself.

For example:

    User
      |
      v
    Finance Model
      |
      +--> Knowledge / RAG
      |
      +--> Calculator
      |
      +--> Financial data tools
      |
      v
    Verified response

---

## 4. Model Strategy

### Model v0 — Qwen3-0.6B

Qwen3-0.6B is the primary research model because:

- Very small parameter count
- Practical for CPU-first experimentation
- Apache 2.0 license
- Official Transformers support
- Small model footprint
- Strong candidate for studying how much capability can be added through targeted fine-tuning

Alternative models will be used later for architectural comparison:

- Llama 3.2 1B Instruct
- Gemma 3 1B IT
- SmolLM2-360M-Instruct

We will NOT fine-tune all models initially.

---

## 5. Training Dataset v0.1

### Financial reasoning

#### FinQA

Official IBM Research dataset.

Focus:

- Financial reports
- Tables
- Numerical reasoning
- Supporting evidence
- Reasoning programs
- Financial QA

FinQA is the primary financial reasoning dataset.

We will preserve its richer information during preprocessing rather than immediately flattening it.

#### TAT-QA

Focus:

- Table + text reasoning
- Financial questions
- Numerical operations
- Multi-step reasoning

TAT-QA complements FinQA.

---

### Accounting knowledge

#### NCERT Accounting Class 11

Approximately 4,408 examples.

Focus:

- Accounting fundamentals
- Accounting process
- Double-entry bookkeeping
- Financial statements
- Accounting principles
- Transactions
- Working capital
- Basic accounting concepts

#### NCERT Accounting Class 12

Approximately 5,730 examples.

Focus:

- More advanced accounting
- Partnership accounting
- Goodwill
- Reconstitution
- Profit-sharing ratios
- Numerical/application questions

Class 11 + Class 12 provide a useful accounting knowledge progression.

---

### Synthetic accounting data

We will create our own synthetic accounting examples with Python.

Advantages:

- Exact ground-truth answers
- Automatically verifiable
- No ambiguity
- Controlled difficulty
- Controlled topic distribution

Potential topics:

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

## 6. Evaluation v0

Evaluation datasets remain strictly separate from training.

### FinQA test

Measures financial numerical reasoning.

### TAT-QA test

Measures table + text financial reasoning.

### Professional Accounting MMLU

Measures accounting knowledge.

### FinanceBench

Measures financial question answering and evidence-based financial reasoning.

### APEX-Accounting

Measures more realistic accounting workflow capability, including tasks such as:

- Reconciliation
- Data entry
- Variance analysis
- Schedules
- Accruals
- Document-heavy accounting work

APEX is evaluation-only.

### Our own accounting benchmark

We will eventually build a controlled benchmark containing synthetic accounting scenarios whose answers can be automatically verified.

---

## 7. Dataset Engineering

We will NOT simply concatenate every dataset.

The preprocessing pipeline will:

1. Inspect schemas
2. Normalize fields
3. Remove duplicates
4. Detect near-duplicates
5. Check answer quality
6. Balance topics
7. Balance difficulty
8. Convert examples into a consistent instruction/chat format
9. Create training and validation splits
10. Protect evaluation data from leakage

The exact dataset proportions will be decided during Dataset v0.1 design.

---

## 8. Planned Experiments

### Experiment 0 — Base Model

Qwen3-0.6B without finance fine-tuning.

Establish baseline performance.

### Experiment 1 — Financial Reasoning

Qwen3-0.6B + selected FinQA/TAT-QA data.

### Experiment 2 — Accounting

Add curated accounting training data.

### Experiment 3 — Synthetic Reasoning

Add verified synthetic accounting calculations.

### Experiment 4 — Combined

Combine the strongest components.

The goal is to determine which data actually improves the model.

---

## 9. Metrics

We will track:

- Answer accuracy
- Numerical accuracy
- Reasoning accuracy
- Accounting classification accuracy
- Hallucination/error rate
- Evaluation score by capability
- RAM usage
- Inference speed
- Training time
- Model size

We should avoid relying on a single overall score.

---

## 10. Deployment Strategy

Ollama is NOT the core training framework.

Current role of Ollama:

- Local inference
- Quick model testing
- Later deployment testing

Training will use the Transformers ecosystem and PEFT/LoRA.

After training, the model can potentially be converted/quantized for GGUF/Ollama deployment.

The friend's online AI platform will be considered later for:

- Larger experiments
- Faster training/inference
- Custom OpenAI-compatible endpoint testing
- Product-oriented experiments

Local research comes first.

---

## 11. Planned Repository Structure

    C:\FinMod
    |
    +-- README.md
    +-- ARCHITECTURE_DECISIONS.md
    |
    +-- docs\
    |   +-- 00-project-goals.md
    |   +-- 01-hardware.md
    |   +-- 02-model-selection.md
    |   +-- 03-dataset-design.md
    |   +-- 04-baseline.md
    |   +-- 05-fine-tuning.md
    |   +-- 06-evaluation.md
    |   +-- 07-results.md
    |
    +-- data\
    |   +-- raw\
    |   +-- processed\
    |   +-- train\
    |   +-- validation\
    |   +-- evaluation\
    |
    +-- src\
    |   +-- data\
    |   +-- training\
    |   +-- evaluation\
    |   +-- inference\
    |
    +-- experiments\
    +-- models\
    +-- notebooks\
    +-- requirements.txt

---

## 12. Current Checkpoint

### Model v0

Qwen3-0.6B — selected.

### Dataset candidates

- FinQA — selected
- TAT-QA — selected
- NCERT Accounting 11 — selected
- NCERT Accounting 12 — selected
- Synthetic accounting — planned

### Evaluation

- FinQA test
- TAT-QA test
- Professional Accounting MMLU
- FinanceBench
- APEX-Accounting
- Our own benchmark

### Deferred

Large generic finance instruction datasets such as 177k/500k datasets are not part of Dataset v0.1 unless later experiments show a clear need.

### Dataset hunting

Paused.

---

## 13. Immediate Next Steps

1. Finalize Dataset v0.1 composition.
2. Define preprocessing format.
3. Define train/validation/evaluation boundaries.
4. Document licensing/provenance.
5. Create the local Python environment.
6. Install training dependencies.
7. Download Qwen3-0.6B.
8. Build the baseline evaluation.
9. Run Experiment 0.
10. Begin the first controlled fine-tuning experiment.

---

## Long-Term Direction

The eventual goal is not simply a chatbot that knows accounting definitions.

The target is a finance/accounting AI that can combine:

- A domain-adapted language model
- Financial reasoning
- Accounting knowledge
- Financial document understanding
- Retrieval/RAG
- Deterministic calculations
- External financial data
- Accounting workflow tools

The project will evolve incrementally toward that architecture.

---

**Project name:** FinMod

**Current version:** Model v0 / Dataset v0.1 design / Evaluation v0

**Status:** Research and architecture phase
