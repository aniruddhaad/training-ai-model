import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"
ADAPTER_PATH = r"C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4"

device = torch.device("cpu")

questions = [
    "What is 2.4 - 2.0?",
    "What is 0.4 / 2.0?",
    "What is 0.2 × 100?",
    "What is (2.4 - 2.0) / 2.0 × 100?",
    
    "Revenue increased from $2 million to $2.4 million. Calculate the percentage increase step by step.",
    "Revenue increased from $3 million to $3.6 million. What is the percentage increase?",
    "Revenue increased from $10 million to $12 million. What is the percentage increase?",
    "Revenue increased from $800,000 to $1,000,000. What is the percentage increase?",
]

def generate(model, tokenizer, question):
    prompt = f"""### Instruction:

{question}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

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


print("=" * 70)
print("ARITHMETIC DIAGNOSTIC")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ---------------------------------------------------------
# BASE MODEL
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASE MODEL")
print("=" * 70)

print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
).to(device)

base_model.eval()

for i, question in enumerate(questions, 1):
    print(f"\n[{i}] {question}")
    print("BASE:", generate(base_model, tokenizer, question))

del base_model

# ---------------------------------------------------------
# LORA MODEL
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASE MODEL + r16 LoRA")
print("=" * 70)

print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
).to(device)

print("Loading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
).to(device)

lora_model.eval()

for i, question in enumerate(questions, 1):
    print(f"\n[{i}] {question}")
    print("LORA:", generate(lora_model, tokenizer, question))

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)