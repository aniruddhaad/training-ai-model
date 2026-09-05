import json
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# ============================================================
# Experiment 9 — Reasoning Training
#
# Base training data:
#   finance_tiny.jsonl
#   finance_tiny_v2.jsonl
#   finance_tiny_v3.jsonl
#   finance_tiny_v4.jsonl
#
# Additional reasoning data:
#   finance_reasoning_v1.jsonl
#
# LoRA:
#   r16
#   QKVO
#   response-only loss
#
# Output:
#   finance_lora_r16_qkvo_response_only_reasoning_v1
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"

DATA_DIR = r"C:\FinMod\data"

BASE_DATASETS = [
    os.path.join(DATA_DIR, "finance_tiny.jsonl"),
    os.path.join(DATA_DIR, "finance_tiny_v2.jsonl"),
    os.path.join(DATA_DIR, "finance_tiny_v3.jsonl"),
    os.path.join(DATA_DIR, "finance_tiny_v4.jsonl"),
]

REASONING_DATASET = os.path.join(
    DATA_DIR,
    "finance_reasoning_v1.jsonl"
)

OUTPUT_DIR = (
    r"C:\FinMod\adapters"
    r"\finance_lora_r16_qkvo_response_only_reasoning_v1"
)


# ------------------------------------------------------------
# Evaluation questions that must NOT enter training
#
# These are the fixed benchmark questions from
# finance_eval_v1.jsonl.
#
# We keep the same exclusion policy used in Exp7.
# ------------------------------------------------------------

EVAL_QUESTIONS_TO_EXCLUDE = {
    "Revenue increased from $2 million to $2.4 million. What is the percentage increase?",
    "A company has revenue of $1,000,000 and gross profit of $400,000. What is its gross margin?",
}


# ------------------------------------------------------------
# Training settings
#
# Keep these aligned with Exp7.
# ------------------------------------------------------------

NUM_EPOCHS = 3

LEARNING_RATE = 2e-4

BATCH_SIZE = 1

MAX_LENGTH = 128

GRADIENT_ACCUMULATION_STEPS = 1

WEIGHT_DECAY = 0.01

LOGGING_STEPS = 1


# ------------------------------------------------------------
# LoRA configuration
#
# Same as Exp7:
#
#   rank        = 16
#   alpha       = 32
#   targets     = Q/K/V/O
#   dropout     = 0.05
#   bias        = none
# ------------------------------------------------------------

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


# ============================================================
# Helper functions
# ============================================================

def load_jsonl(path):
    """Load JSONL examples from a file."""

    examples = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {e}"
                )

    return examples


def build_prompt(question):
    """
    Use the same prompt format as previous experiments.
    """

    return (
        "### Instruction:\n"
        f"{question}\n\n"
        "### Response:\n"
    )


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("Experiment 9 — LoRA Reasoning Training")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device        : {device}")
print(f"Base model    : {MODEL_PATH}")
print(f"Output        : {OUTPUT_DIR}")
print()


# ============================================================
# Verify files
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Base model not found:\n{MODEL_PATH}"
    )


for path in BASE_DATASETS:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Training dataset not found:\n{path}"
        )


if not os.path.exists(REASONING_DATASET):

    raise FileNotFoundError(
        f"Reasoning dataset not found:\n{REASONING_DATASET}"
    )


# ============================================================
# Load datasets
# ============================================================

print("Loading training datasets...")
print()

all_examples = []

for path in BASE_DATASETS:

    examples = load_jsonl(path)

    print(
        f"{os.path.basename(path):35s} "
        f"{len(examples):3d} examples"
    )

    all_examples.extend(examples)


print()

reasoning_examples = load_jsonl(
    REASONING_DATASET
)

print(
    f"{os.path.basename(REASONING_DATASET):35s} "
    f"{len(reasoning_examples):3d} examples"
)

print()


# ============================================================
# Exclude fixed evaluation questions
# ============================================================

before_exclusion = len(all_examples)

all_examples = [
    example
    for example in all_examples
    if example.get("instruction", "")
    not in EVAL_QUESTIONS_TO_EXCLUDE
]


excluded_count = (
    before_exclusion - len(all_examples)
)

print("Base dataset:")
print(f"  Before exclusion : {before_exclusion}")
print(f"  Excluded         : {excluded_count}")
print(f"  After exclusion  : {len(all_examples)}")
print()


# ============================================================
# IMPORTANT:
#
# finance_reasoning_v1.jsonl is additional TRAINING data.
#
# The clean reasoning evaluation file is NOT loaded here.
# ============================================================

print("Adding reasoning training examples...")

all_examples.extend(reasoning_examples)

print(
    f"Total training examples : {len(all_examples)}"
)

print()


# ============================================================
# Dataset sanity checks
# ============================================================

print("Dataset sanity checks...")

question_keys = []

for example in all_examples:

    if "instruction" in example:

        question = example["instruction"]

    elif "question" in example:

        question = example["question"]

    else:

        raise ValueError(
            "Training example does not contain "
            "'instruction' or 'question'."
        )

    if "response" not in example:

        raise ValueError(
            f"Training example has no response:\n{example}"
        )

    question_keys.append(question)


if len(question_keys) != len(set(question_keys)):

    print(
        "WARNING: Duplicate training questions detected."
    )

else:

    print("  No duplicate training questions detected.")


print()


# ============================================================
# Build text examples
#
# Response-only loss:
#
#   ### Instruction:
#   question
#
#   ### Response:
#   answer
#
# Only tokens AFTER the response marker receive loss.
# ============================================================

print("Building training texts...")

training_texts = []

for example in all_examples:

    if "instruction" in example:

        question = example["instruction"]

    else:

        question = example["question"]

    response = example["response"]

    prompt = build_prompt(question)

    full_text = prompt + response

    training_texts.append(full_text)


print(
    f"Prepared {len(training_texts)} training examples."
)

print()


# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print()


# ============================================================
# Tokenize dataset
# ============================================================

print("Tokenizing dataset...")

tokenized_examples = []

response_marker = "### Response:\n"

for text in training_texts:

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = encoded["input_ids"]

    attention_mask = encoded["attention_mask"]

    labels = input_ids.copy()

    # --------------------------------------------------------
    # Response-only loss
    #
    # Find the response marker and mask everything before it.
    # --------------------------------------------------------

    marker_ids = tokenizer(
        response_marker,
        add_special_tokens=False
    )["input_ids"]

    response_start = None

    for i in range(
        0,
        len(input_ids) - len(marker_ids) + 1
    ):

        if input_ids[
            i:i + len(marker_ids)
        ] == marker_ids:

            response_start = (
                i + len(marker_ids)
            )

            break

    if response_start is None:

        raise ValueError(
            "Could not locate response marker "
            "during tokenization."
        )

    # Mask prompt tokens.
    for i in range(response_start):

        labels[i] = -100

    tokenized_examples.append({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    })


print("Tokenization complete.")
print()


# ============================================================
# Simple Dataset class
# ============================================================

class FinanceDataset(torch.utils.data.Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


train_dataset = FinanceDataset(
    tokenized_examples
)


# ============================================================
# Load base model
# ============================================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
)

print("Base model loaded.")
print()


# ============================================================
# Attach LoRA
# ============================================================

print("Attaching LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()

print()


# ============================================================
# Training arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_steps=0,

    logging_steps=LOGGING_STEPS,

    save_strategy="no",

    report_to="none",

    fp16=False,

    bf16=False,

    dataloader_pin_memory=False,

    remove_unused_columns=False,
)


# ============================================================
# Data collator
#
# We already created labels manually, so this collator only
# needs to pad the batch.
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    data_collator=data_collator,
)


# ============================================================
# Train
# ============================================================

print("=" * 70)
print("Starting Experiment 9 training")
print("=" * 70)

print()
print(f"Training examples : {len(train_dataset)}")
print(f"Epochs            : {NUM_EPOCHS}")
print(f"Learning rate     : {LEARNING_RATE}")
print(f"LoRA rank         : {LORA_R}")
print(f"LoRA alpha        : {LORA_ALPHA}")
print(f"Target modules    : {', '.join(TARGET_MODULES)}")
print()
print("Training...")
print()


train_result = trainer.train()


# ============================================================
# Training summary
# ============================================================

print()
print("=" * 70)
print("Training complete")
print("=" * 70)

print()

print("Training metrics:")

for key, value in train_result.metrics.items():

    print(
        f"  {key}: {value}"
    )

print()


# ============================================================
# Save adapter
# ============================================================

print("Saving LoRA adapter...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    f"Adapter saved to:\n{OUTPUT_DIR}"
)

print()


# ============================================================
# Final summary
# ============================================================

print("=" * 70)
print("Experiment 9 finished successfully")
print("=" * 70)

print()
print("Training data:")
print("  finance_tiny.jsonl")
print("  finance_tiny_v2.jsonl")
print("  finance_tiny_v3.jsonl")
print("  finance_tiny_v4.jsonl")
print("  finance_reasoning_v1.jsonl")
print()

print(
    f"Total training examples: {len(train_dataset)}"
)

print()
print("LoRA:")
print(f"  Rank          : {LORA_R}")
print(f"  Alpha         : {LORA_ALPHA}")
print("  Targets       : Q/K/V/O")
print("  Loss          : Response-only")

print()
print("Next step:")
print(
    "Run the SAME 20-question benchmark "
    "against this new adapter."
)

print()
print(
    r"Expected adapter:"
)
print(
    OUTPUT_DIR
)

print("=" * 70)