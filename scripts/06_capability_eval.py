import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"
ADAPTER_PATH = r"C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4"

device = torch.device("cpu")


tests = [

    # =====================================================
    # A. FINANCE CONCEPTS
    # =====================================================

    (
        "A1",
        "Finance Concepts",
        "What is revenue?"
    ),

    (
        "A2",
        "Finance Concepts",
        "What is the difference between an asset and a liability?"
    ),

    (
        "A3",
        "Finance Concepts",
        "What is gross profit?"
    ),

    (
        "A4",
        "Finance Concepts",
        "What is net profit?"
    ),

    (
        "A5",
        "Finance Concepts",
        "Why can a profitable company have negative operating cash flow?"
    ),


    # =====================================================
    # B. FINANCE FORMULAS
    # =====================================================

    (
        "B1",
        "Finance Formulas",
        "What is the formula for working capital?"
    ),

    (
        "B2",
        "Finance Formulas",
        "A company has current assets of $300,000 and current liabilities of $180,000. What is its working capital?"
    ),

    (
        "B3",
        "Finance Formulas",
        "A company has revenue of $2,000,000 and gross profit of $500,000. What is its gross margin?"
    ),

    (
        "B4",
        "Finance Formulas",
        "A company's revenue increases from $4 million to $5 million. What is the percentage increase?"
    ),

    (
        "B5",
        "Finance Formulas",
        "A company's revenue decreases from $5 million to $4 million. What is the percentage decrease?"
    ),


    # =====================================================
    # C. ACCOUNTING TRANSACTIONS
    # =====================================================

    (
        "C1",
        "Accounting Transactions",
        "A company receives a $100,000 bank loan in cash. What happens to assets, liabilities, and equity?"
    ),

    (
        "C2",
        "Accounting Transactions",
        "An owner invests $100,000 cash into the company. What happens to assets, liabilities, and equity?"
    ),

    (
        "C3",
        "Accounting Transactions",
        "A company repays $30,000 of a bank loan using cash. What happens to assets, liabilities, and equity?"
    ),

    (
        "C4",
        "Accounting Transactions",
        "A company purchases equipment for $40,000 cash. What happens to total assets, liabilities, and equity?"
    ),

    (
        "C5",
        "Accounting Transactions",
        "A company earns $50,000 of revenue in cash and has no related expenses. What happens to assets, liabilities, and equity?"
    ),


    # =====================================================
    # D. ARITHMETIC
    # =====================================================

    (
        "D1",
        "Arithmetic",
        "What is 17 + 28?"
    ),

    (
        "D2",
        "Arithmetic",
        "What is 125 - 47?"
    ),

    (
        "D3",
        "Arithmetic",
        "What is 24 × 15?"
    ),

    (
        "D4",
        "Arithmetic",
        "What is 144 / 12?"
    ),

    (
        "D5",
        "Arithmetic",
        "What is 0.35 × 100?"
    ),

    (
        "D6",
        "Arithmetic",
        "What is 1.2 - 0.75?"
    ),


    # =====================================================
    # E. MULTI-STEP FINANCE REASONING
    # =====================================================

    (
        "E1",
        "Multi-Step Reasoning",
        "A company has revenue of $1,000,000 and cost of goods sold of $600,000. It has operating expenses of $250,000. What are its gross profit and operating profit?"
    ),

    (
        "E2",
        "Multi-Step Reasoning",
        "A company has current assets of $500,000 and current liabilities of $350,000. It pays $50,000 of accounts payable using cash. What is its new working capital?"
    ),

    (
        "E3",
        "Multi-Step Reasoning",
        "A company reports net income of $200,000 and depreciation expense of $30,000. Under the indirect method, what is the operating cash flow before considering other adjustments?"
    ),

    (
        "E4",
        "Multi-Step Reasoning",
        "A company records $100,000 of credit sales. Accounts receivable increases by $100,000 and cash is not collected. What is the effect on operating cash flow?"
    ),

    (
        "E5",
        "Multi-Step Reasoning",
        "A company's revenue increases from $2 million to $2.5 million. Its cost of goods sold increases from $1.2 million to $1.5 million. What happens to gross profit?"
    ),
]


def generate(model, tokenizer, question):

    prompt = f"""### Instruction:

{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
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


print("=" * 80)
print("FINANCE MODEL CAPABILITY EVALUATION")
print("=" * 80)

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


current_category = None

for test_id, category, question in tests:

    if category != current_category:
        print("\n" + "=" * 80)
        print(category.upper())
        print("=" * 80)
        current_category = category

    print(f"\n[{test_id}] {question}")
    print("ANSWER:")
    print(generate(model, tokenizer, question))


print("\n" + "=" * 80)
print("CAPABILITY EVALUATION COMPLETE")
print("=" * 80)