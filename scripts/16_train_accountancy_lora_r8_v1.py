"""
15_train_accountancy_lora_response_only.py

Production v1 accounting-training pipeline for C:\FinMod.

IMPORTANT:
- Uses the cleaned Class 11 and Class 12 CSV datasets.
- Uses a CUSTOM collator. It does NOT use
  DataCollatorForLanguageModeling, because that collator overwrites manually
  masked labels when mlm=False.
- Loss is response-only:
      prompt/question tokens -> -100
      answer tokens          -> actual token IDs
- Creates a deterministic train/validation split separately within each class.
- Includes a sanity mode that performs only a few optimizer steps before the
  full run.

Usage:

1) Sanity check (recommended first):
    cd C:\FinMod
    .\.venv\Scripts\Activate.ps1
    python scripts\15_train_accountancy_lora_response_only.py --sanity

2) Full training:
    python scripts\15_train_accountancy_lora_response_only.py

Production v1 defaults: LoRA r=8, q/k/v/o, alpha=16, max_length=256,
1 epoch. Change constants below only deliberately.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"C:\FinMod")
MODEL_PATH = BASE_DIR / "models" / "SmolLM2-360M"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
ADAPTER_DIR = BASE_DIR / "adapters" / "accountancy_lora_r8_qkvo_response_only_v1"

CLASS11_PATH = DATA_DIR / "Accountancy_11th_FINAL.csv"
CLASS12_PATH = DATA_DIR / "Accountancy_12th_FINAL.csv"

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED = 42
VALIDATION_FRACTION = 0.05

MAX_LENGTH = 256

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

# CPU machine: keep this conservative.
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 25
SAVE_STEPS = 500
EVAL_STEPS = 500

# Sanity run: enough steps to prove the pipeline actually trains.
SANITY_MAX_STEPS = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_csv(path, class_name):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"Question", "Answer", "Topic"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    cleaned = []
    for row in rows:
        question = clean_text(row.get("Question"))
        answer = clean_text(row.get("Answer"))

        if not question or not answer:
            continue

        cleaned.append({
            "class": class_name,
            "topic": clean_text(row.get("Topic")),
            "difficulty": clean_text(row.get("Difficulty")),
            "student_level": clean_text(row.get("StudentLevel")),
            "question_type": clean_text(row.get("QuestionType")),
            "question_complexity": clean_text(row.get("QuestionComplexity")),
            "prerequisites": clean_text(row.get("Prerequisites")),
            "estimated_time": clean_text(row.get("EstimatedTime")),
            "subject": clean_text(row.get("subject")),
            "grade": clean_text(row.get("grade")),
            "question": question,
            "answer": answer,
        })

    return cleaned


def deduplicate(rows):
    """Remove only exact question+answer duplicates."""
    seen = set()
    result = []

    for row in rows:
        key = (row["question"].lower(), row["answer"].lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(row)

    return result


def split_by_class(rows, validation_fraction, seed):
    """
    Deterministic split performed separately for Class 11 and Class 12 so both
    classes are represented in train and validation sets.
    """
    rng = random.Random(seed)

    by_class = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)

    train = []
    validation = []

    for class_name in sorted(by_class):
        items = list(by_class[class_name])
        rng.shuffle(items)

        n_val = max(1, round(len(items) * validation_fraction))
        validation.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(validation)

    return train, validation


def build_prompt(question):
    return (
        "You are an accounting tutor. Answer the following accountancy "
        "question clearly and accurately.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_example(row, tokenizer):
    """
    Tokenize prompt and answer separately so the exact prompt/answer boundary
    is known. Prompt labels are -100; answer labels are trainable.

    EOS is appended to the answer so the model also learns where an answer
    should terminate.
    """
    prompt = build_prompt(row["question"])

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    answer_ids = tokenizer(
        " " + row["answer"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]

    # Reserve at least one answer token. If the combined sequence is too long,
    # keep the full answer as much as possible and truncate the prompt first.
    if len(prompt_ids) + len(answer_ids) > MAX_LENGTH:
        available_prompt = MAX_LENGTH - len(answer_ids)

        if available_prompt < 1:
            # Extremely long answer: truncate answer while preserving the
            # response-only objective.
            answer_ids = answer_ids[:MAX_LENGTH]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[:available_prompt]

    input_ids = prompt_ids + answer_ids
    labels = ([-100] * len(prompt_ids)) + answer_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# CUSTOM COLLATOR
# ---------------------------------------------------------------------------

class ResponseOnlyCollator:
    """
    Pads input_ids/attention_mask/labels while preserving the labels supplied
    by tokenization.

    This is the critical fix. Do NOT replace this with
    DataCollatorForLanguageModeling(mlm=False).
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(
                x["input_ids"] + [self.pad_token_id] * pad_len
            )
            attention_mask.append(
                x["attention_mask"] + [0] * pad_len
            )
            labels.append(
                x["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AccountancyDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

        self.items = [
            tokenize_example(row, tokenizer)
            for row in rows
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def dataset_stats(dataset):
    total_tokens = 0
    trainable_tokens = 0

    for item in dataset.items:
        total_tokens += len(item["input_ids"])
        trainable_tokens += sum(
            x != -100 for x in item["labels"]
        )

    return {
        "examples": len(dataset),
        "total_tokens": total_tokens,
        "trainable_answer_tokens": trainable_tokens,
        "avg_total_tokens": (
            total_tokens / len(dataset) if len(dataset) else 0
        ),
        "avg_answer_tokens": (
            trainable_tokens / len(dataset) if len(dataset) else 0
        ),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_masking(dataset):
    """Fail fast if prompt tokens are not masked."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    for i in range(min(10, len(dataset))):
        item = dataset[i]

        first_trainable = None
        for j, label in enumerate(item["labels"]):
            if label != -100:
                first_trainable = j
                break

        if first_trainable is None:
            raise RuntimeError(
                f"Example {i} has no trainable answer tokens."
            )

        if any(label != -100 for label in item["labels"][:first_trainable]):
            raise RuntimeError(
                f"Example {i} has unmasked prompt tokens."
            )

    print("\nMASKING CHECK: PASS")
    print("Prompt labels are -100; answer labels are trainable.")


def verify_collator(dataset, tokenizer):
    """Verify padding also preserves response-only labels."""
    sample_count = min(2, len(dataset))
    features = [dataset[i] for i in range(sample_count)]

    collator = ResponseOnlyCollator(tokenizer.pad_token_id)
    batch = collator(features)

    for row_idx in range(sample_count):
        original = features[row_idx]
        padded_labels = batch["labels"][row_idx].tolist()

        original_len = len(original["labels"])
        if padded_labels[:original_len] != original["labels"]:
            raise RuntimeError(
                "CUSTOM COLLATOR CHANGED EXISTING LABELS."
            )

        if any(x != -100 for x in padded_labels[original_len:]):
            raise RuntimeError(
                "CUSTOM COLLATOR FAILED TO MASK padding labels."
            )

    print("COLLATOR CHECK: PASS")
    print("Custom collator preserves labels and masks padding.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run only a 5-step training sanity check.",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 78)
    print("ACCOUNTANCY LORA TRAINING - RESPONSE-ONLY LOSS")
    print("=" * 78)
    print(f"Model: {MODEL_PATH}")
    print(f"Class 11: {CLASS11_PATH}")
    print(f"Class 12: {CLASS12_PATH}")
    print(f"Sanity mode: {args.sanity}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS11_PATH.exists():
        raise FileNotFoundError(f"Class 11 dataset not found: {CLASS11_PATH}")

    if not CLASS12_PATH.exists():
        raise FileNotFoundError(f"Class 12 dataset not found: {CLASS12_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------- Load and prepare data --------------------

    rows11 = load_csv(CLASS11_PATH, "Class11")
    rows12 = load_csv(CLASS12_PATH, "Class12")

    print("\nSOURCE DATA")
    print("-" * 78)
    print(f"Class 11 usable rows: {len(rows11)}")
    print(f"Class 12 usable rows: {len(rows12)}")

    rows = deduplicate(rows11 + rows12)

    print(f"Combined after exact Q+A deduplication: {len(rows)}")

    train_rows, val_rows = split_by_class(
        rows,
        VALIDATION_FRACTION,
        SEED,
    )

    print("\nSPLIT")
    print("-" * 78)
    print(f"Training rows:   {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")

    print("Training class distribution:")
    print(Counter(x["class"] for x in train_rows))

    print("Validation class distribution:")
    print(Counter(x["class"] for x in val_rows))

    # -------------------- Tokenizer --------------------

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"PAD token: {tokenizer.pad_token!r}")
    print(f"EOS token: {tokenizer.eos_token!r}")

    # -------------------- Tokenize --------------------

    print("\nTokenizing datasets...")
    train_dataset = AccountancyDataset(train_rows, tokenizer)
    val_dataset = AccountancyDataset(val_rows, tokenizer)

    train_stats = dataset_stats(train_dataset)
    val_stats = dataset_stats(val_dataset)

    print("\nTOKENIZATION STATS")
    print("-" * 78)
    print(json.dumps({
        "train": train_stats,
        "validation": val_stats,
    }, indent=2))

    verify_masking(train_dataset)
    verify_masking(val_dataset)
    verify_collator(train_dataset, tokenizer)

    # -------------------- Manifest --------------------

    manifest = {
        "experiment": "Accountancy LoRA response-only",
        "model_path": str(MODEL_PATH),
        "class11_source": str(CLASS11_PATH),
        "class12_source": str(CLASS12_PATH),
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "max_length": MAX_LENGTH,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "per_device_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        },
        "loss": "response_only",
        "collator": "custom ResponseOnlyCollator",
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_stats": train_stats,
        "validation_stats": val_stats,
        "sanity_mode": args.sanity,
    }

    manifest_path = OUTPUT_DIR / "accountancy_lora_response_only_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------- Model + LoRA --------------------

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("\nLoRA parameters:")
    model.print_trainable_parameters()

    # -------------------- Trainer --------------------

    run_dir = (
        ADAPTER_DIR / "sanity"
        if args.sanity
        else ADAPTER_DIR / "full"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Transformers 5.x in this environment does not accept
    # overwrite_output_dir. Existing output directories are handled by the
    # explicit run directories below.
    training_args = TrainingArguments(
        output_dir=str(run_dir),

        num_train_epochs=NUM_EPOCHS,
        max_steps=SANITY_MAX_STEPS if args.sanity else -1,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,
        optim="adamw_torch",

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,

        eval_strategy="steps",
        save_strategy="steps",

        save_total_limit=2,
        report_to="none",

        fp16=False,
        bf16=False,

        seed=SEED,
        data_seed=SEED,

        remove_unused_columns=False,
    )

    collator = ResponseOnlyCollator(tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # -------------------- Final pre-flight --------------------

    print("\nPRE-FLIGHT")
    print("-" * 78)
    print("Custom response-only collator: ENABLED")
    print("Prompt loss: MASKED (-100)")
    print("Answer loss: ENABLED")
    print("DataCollatorForLanguageModeling: NOT USED")

    if args.sanity:
        print("\nStarting 5-step SANITY training run...")
    else:
        print("\nStarting FULL training run...")

    # -------------------- Train --------------------

    result = trainer.train()

    print("\nTRAINING COMPLETE")
    print("-" * 78)
    print(result)

    # -------------------- Save adapter --------------------

    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save final training metrics.
    metrics_path = run_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nAdapter saved to: {final_dir}")
    print(f"Manifest:         {manifest_path}")
    print(f"Metrics:          {metrics_path}")

    if args.sanity:
        print("\nSANITY RUN FINISHED.")
        print("Inspect the loss and saved adapter before running the full job.")


if __name__ == "__main__":
    main()
