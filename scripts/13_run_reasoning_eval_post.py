import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Experiment 9 — POST-Training Reasoning Evaluation
#
# Purpose:
#   Run the exact same clean reasoning benchmark used for the
#   PRE-training evaluation, but against the Experiment 9
#   reasoning-trained adapter.
#
# Base model:
#   SmolLM2-360M
#
# POST adapter:
#   r16 QKVO response-only + reasoning_v1
#
# Evaluation:
#   finance_reasoning_eval_v1.jsonl
#
# Output:
#   finance_reasoning_eval_v1_post.jsonl
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"

ADAPTER_PATH = (
    r"C:\FinMod\adapters"
    r"\finance_lora_r16_qkvo_response_only_reasoning_v1"
)

EVAL_PATH = r"C:\FinMod\data\finance_reasoning_eval_v1.jsonl"

OUTPUT_PATH = (
    r"C:\FinMod\outputs"
    r"\finance_reasoning_eval_v1_post.jsonl"
)


# ------------------------------------------------------------
# Generation settings
#
# MUST remain identical to script 11.
# ------------------------------------------------------------

MAX_NEW_TOKENS = 60


# ------------------------------------------------------------
# Stop markers
#
# MUST remain identical to script 11.
# ------------------------------------------------------------

STOP_MARKERS = [
    "### Instruction:",
    "### Question:",
    "### Response:",
    "Instruction:",
    "Question:",
    "Response:",
]


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("Experiment 9 — POST-Training Reasoning Evaluation")
print("=" * 70)

print(f"Device       : {device}")
print(f"Base model   : {MODEL_PATH}")
print(f"Adapter      : {ADAPTER_PATH}")
print(f"Evaluation   : {EVAL_PATH}")
print(f"Output       : {OUTPUT_PATH}")
print()


# ------------------------------------------------------------
# Check paths
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Base model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(
        f"Experiment 9 adapter not found:\n{ADAPTER_PATH}"
    )

if not os.path.exists(EVAL_PATH):
    raise FileNotFoundError(
        f"Evaluation dataset not found:\n{EVAL_PATH}"
    )


# ------------------------------------------------------------
# Load evaluation dataset
# ------------------------------------------------------------

eval_data = []

with open(EVAL_PATH, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        eval_data.append(
            json.loads(line)
        )


print(
    f"Loaded {len(eval_data)} evaluation examples."
)

if len(eval_data) != 20:

    print(
        f"WARNING: Expected 20 examples, "
        f"but found {len(eval_data)}."
    )

print()


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print()


# ------------------------------------------------------------
# Load base model
# ------------------------------------------------------------

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
)

model.to(device)

print("Base model loaded.")
print()


# ------------------------------------------------------------
# Load Experiment 9 adapter
# ------------------------------------------------------------

print("Loading Experiment 9 LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.to(device)

model.eval()

print("Experiment 9 adapter loaded.")
print()


# ------------------------------------------------------------
# Model configuration
#
# During evaluation no parameters should be trainable.
# ------------------------------------------------------------

print("Model configuration:")

try:
    model.print_trainable_parameters()
except Exception:
    pass

print()


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------

results = []

print("=" * 70)
print("Running POST-training evaluation")
print("=" * 70)
print()


for index, item in enumerate(
    eval_data,
    start=1
):

    question = item["question"]

    # --------------------------------------------------------
    # EXACT SAME PROMPT FORMAT AS PRE EVALUATION
    # --------------------------------------------------------

    prompt = (
        "### Instruction:\n"
        f"{question}\n\n"
        "### Response:\n"
    )

    print(
        f"[{index}/{len(eval_data)}] {item['id']}"
    )

    print(
        f"Category : {item['category']}"
    )

    print(
        f"Question : {question}"
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Deterministic generation
    #
    # EXACT SAME SETTINGS AS PRE EVALUATION.
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # --------------------------------------------------------
    # Decode ONLY newly generated tokens
    # --------------------------------------------------------

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated_ids = output_ids[
        0,
        input_length:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------------
    # Remove accidental prompt continuation
    # --------------------------------------------------------

    for marker in STOP_MARKERS:

        if marker in answer:

            answer = answer.split(
                marker
            )[0].strip()

    print(
        f"Answer   : {answer}"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "expected_answer": item["expected_answer"],
        "model_answer": answer,
    })


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for result in results:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("POST-training evaluation complete")
print("=" * 70)

print(
    f"Examples evaluated : {len(results)}"
)

print(
    f"Results saved to   : {OUTPUT_PATH}"
)

print()
print("IMPORTANT:")
print(
    "This is the POST-training evaluation for Experiment 9."
)

print(
    "The exact same 20-question benchmark was used for "
    "the PRE evaluation."
)

print()
print("Next step:")
print(
    "Compare finance_reasoning_eval_v1_pre.jsonl "
    "against finance_reasoning_eval_v1_post.jsonl."
)

print("=" * 70)