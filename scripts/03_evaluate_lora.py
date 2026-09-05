import gc
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"
ADAPTER_PATH = r"C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4"

device = torch.device("cpu")


# --------------------------------------------------
# Evaluation questions
# --------------------------------------------------

questions = [
    "What is the difference between gross profit and net profit?",
    "A company has current assets of $200,000 and current liabilities of $125,000. What is its working capital?",
    "Revenue increased from $2 million to $2.4 million. What is the percentage increase?",
    "Why can an increase in accounts receivable reduce operating cash flow?",
    "A company takes a $50,000 bank loan. What happens to the accounting equation?",
    "If a company has revenue of $1,000,000 and gross profit of $400,000, what is its gross margin?",
    "Why is depreciation added back when calculating operating cash flow under the indirect method?",
]


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

def generate_answer(model, question):

    prompt = (
        "### Instruction:\n"
        + question
        + "\n\n"
        + "### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Only keep newly generated tokens
    generated_ids = output_ids[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------
    # Remove training-format leakage
    # --------------------------------------------------

    stop_markers = [
        "### Instruction:",
        "### Question:",
        "### Response:",
        "Instruction:",
        "Question:",
        "Response:",
    ]

    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    return answer


# ==================================================
# BASE MODEL
# ==================================================

print("=" * 70)
print("BASE MODEL")
print("=" * 70)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32
)

base_model.to(device)
base_model.eval()

base_answers = []

for i, question in enumerate(questions, start=1):

    print()
    print(f"Question {i}: {question}")

    answer = generate_answer(
        base_model,
        question
    )

    base_answers.append(answer)

    print(f"Base answer: {answer}")


# --------------------------------------------------
# Free memory
# --------------------------------------------------

del base_model
gc.collect()


# ==================================================
# BASE + LoRA
# ==================================================

print()
print("=" * 70)
print("BASE MODEL + LoRA ADAPTER")
print("=" * 70)

print("Loading base model...")

lora_base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32
)

lora_base.to(device)

print("Loading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    lora_base,
    ADAPTER_PATH
)

lora_model.to(device)
lora_model.eval()

lora_answers = []

for i, question in enumerate(questions, start=1):

    print()
    print(f"Question {i}: {question}")

    answer = generate_answer(
        lora_model,
        question
    )

    lora_answers.append(answer)

    print(f"LoRA answer: {answer}")


# ==================================================
# FINAL COMPARISON
# ==================================================

print()
print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

for i, question in enumerate(questions):

    print()
    print(f"[{i + 1}] {question}")

    print()
    print("BASE:")
    print(base_answers[i])

    print()
    print("LORA:")
    print(lora_answers[i])

    print("-" * 70)


print()
print("Evaluation complete.")