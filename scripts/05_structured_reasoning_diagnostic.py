import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"
ADAPTER_PATH = r"C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4"

device = torch.device("cpu")

tests = [
    (
        "DIRECT — $3M to $3.6M",
        """Revenue increased from $3 million to $3.6 million.
What is the percentage increase?"""
    ),
    (
        "STRUCTURED — $3M to $3.6M",
        """Old revenue = $3 million.
New revenue = $3.6 million.

Step 1: Calculate the increase.
Step 2: Divide the increase by the old revenue.
Step 3: Multiply by 100.
Give the final percentage."""
    ),

    (
        "DIRECT — $10M to $12M",
        """Revenue increased from $10 million to $12 million.
What is the percentage increase?"""
    ),
    (
        "STRUCTURED — $10M to $12M",
        """Old revenue = $10 million.
New revenue = $12 million.

Step 1: Calculate the increase.
Step 2: Divide the increase by the old revenue.
Step 3: Multiply by 100.
Give the final percentage."""
    ),

    (
        "DIRECT — $800K to $1M",
        """Revenue increased from $800,000 to $1,000,000.
What is the percentage increase?"""
    ),
    (
        "STRUCTURED — $800K to $1M",
        """Old revenue = $800,000.
New revenue = $1,000,000.

Step 1: Calculate the increase.
Step 2: Divide the increase by the old revenue.
Step 3: Multiply by 100.
Give the final percentage."""
    ),

    (
        "STRUCTURED — completely new numbers",
        """Old revenue = $4 million.
New revenue = $5 million.

Step 1: Calculate the increase.
Step 2: Divide the increase by the old revenue.
Step 3: Multiply by 100.
Give the final percentage."""
    ),
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
            max_new_tokens=100,
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
print("STRUCTURED REASONING DIAGNOSTIC")
print("=" * 70)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
).to(device)

print("Loading r16 LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
).to(device)

model.eval()

for i, (name, question) in enumerate(tests, 1):
    print("\n" + "-" * 70)
    print(f"[{i}] {name}")
    print("-" * 70)
    print(question)
    print("\nLORA:")
    print(generate(model, tokenizer, question))

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)