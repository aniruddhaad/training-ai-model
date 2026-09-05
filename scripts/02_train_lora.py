import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"

DATA_PATH_1 = r"C:\FinMod\data\finance_tiny.jsonl"
DATA_PATH_2 = r"C:\FinMod\data\finance_tiny_v2.jsonl"
DATA_PATH_3 = r"C:\FinMod\data\finance_tiny_v3.jsonl"
DATA_PATH_4 = r"C:\FinMod\data\finance_tiny_v4.jsonl"

# New adapter for Experiment 4
OUTPUT_PATH = r"C:\FinMod\adapters\finance_lora_r16_qkvo_response_only_v4"

device = torch.device("cpu")


# --------------------------------------------------
# 1. Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# 2. Load base model
# --------------------------------------------------

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32
)

model.to(device)


# --------------------------------------------------
# 3. Attach LoRA
# --------------------------------------------------

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

model.print_trainable_parameters()


# --------------------------------------------------
# 4. Load both datasets
# --------------------------------------------------

print()
print("Loading datasets...")

examples = []


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


examples_1 = load_jsonl(DATA_PATH_1)
examples_2 = load_jsonl(DATA_PATH_2)
examples_3 = load_jsonl(DATA_PATH_3)
examples_4 = load_jsonl(DATA_PATH_4)

print(f"Dataset 1: {len(examples_1)} examples")
print(f"Dataset 2: {len(examples_2)} examples")
print(f"Dataset 3: {len(examples_3)} examples")
print(f"Dataset 4: {len(examples_4)} examples")


# --------------------------------------------------
# 5. Combine datasets
# --------------------------------------------------

examples = examples_1 + examples_2 + examples_3 + examples_4

print(f"Combined dataset: {len(examples)} examples")


# --------------------------------------------------
# 6. Remove exact evaluation examples
# --------------------------------------------------
#
# These questions must remain unseen during training.
#
# Q3:
# Revenue increased from $2 million to $2.4 million.
#
# Q6:
# Revenue = $1,000,000
# Gross profit = $400,000
#
# These exact examples exist in finance_tiny_v2.jsonl.
#
# We remove them so the evaluation remains meaningful.
# --------------------------------------------------

EVAL_QUESTIONS_TO_EXCLUDE = {
    "Revenue increased from $2 million to $2.4 million. What is the percentage increase?",
    "A company has revenue of $1,000,000 and gross profit of $400,000. What is its gross margin?"
}


before_filter = len(examples)

examples = [
    example
    for example in examples
    if example["instruction"] not in EVAL_QUESTIONS_TO_EXCLUDE
]

removed_count = before_filter - len(examples)

print(f"Excluded evaluation examples: {removed_count}")
print(f"Final training dataset: {len(examples)} examples")


# --------------------------------------------------
# 7. Tokenize with response-only labels
# --------------------------------------------------

MAX_LENGTH = 128

tokenized = []

for example in examples:

    instruction_text = (
        "### Instruction:\n"
        + example["instruction"]
        + "\n\n"
        + "### Response:\n"
    )

    response_text = example["response"]


    # --------------------------------------------------
    # Tokenize instruction separately
    # --------------------------------------------------

    instruction_tokens = tokenizer(
        instruction_text,
        add_special_tokens=True,
        truncation=False
    )


    # --------------------------------------------------
    # Tokenize response separately
    # --------------------------------------------------

    response_tokens = tokenizer(
        response_text,
        add_special_tokens=False,
        truncation=False
    )


    # --------------------------------------------------
    # Build input
    # --------------------------------------------------

    input_ids = (
        instruction_tokens["input_ids"]
        + response_tokens["input_ids"]
    )

    # Add EOS to mark end of response
    input_ids.append(tokenizer.eos_token_id)


    # --------------------------------------------------
    # Build response-only labels
    #
    # -100 = ignore when calculating loss
    #
    # Instruction → ignored
    # Response    → trained
    # --------------------------------------------------

    instruction_length = len(
        instruction_tokens["input_ids"]
    )

    labels = (
        [-100] * instruction_length
        + response_tokens["input_ids"]
        + [tokenizer.eos_token_id]
    )


    # --------------------------------------------------
    # Truncate
    # --------------------------------------------------

    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]


    # --------------------------------------------------
    # Attention mask
    # --------------------------------------------------

    attention_mask = [1] * len(input_ids)


    # --------------------------------------------------
    # Padding
    # --------------------------------------------------

    padding_length = MAX_LENGTH - len(input_ids)

    if padding_length > 0:

        input_ids += [
            tokenizer.pad_token_id
        ] * padding_length

        attention_mask += [0] * padding_length

        labels += [-100] * padding_length


    tokenized.append({
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long
        )
    })


print(f"Tokenized {len(tokenized)} examples.")


# --------------------------------------------------
# 8. Check response-only loss masking
# --------------------------------------------------

print()
print("Checking response-only loss masking...")

sample = tokenized[0]

decoded_input = tokenizer.decode(
    sample["input_ids"],
    skip_special_tokens=True
)

print()
print("Example input:")
print(decoded_input)

print()
print("Tokens contributing to loss:")

loss_token_ids = sample["labels"][
    sample["labels"] != -100
]

print(
    tokenizer.decode(
        loss_token_ids,
        skip_special_tokens=True
    )
)


# --------------------------------------------------
# 9. Optimizer
# --------------------------------------------------

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=2e-4
)


# --------------------------------------------------
# 10. Training
# --------------------------------------------------

model.train()

EPOCHS = 3

print()
print("Starting training...")
print("--------------------")

for epoch in range(EPOCHS):

    total_loss = 0.0

    for step, batch in enumerate(tokenized):

        input_ids = (
            batch["input_ids"]
            .unsqueeze(0)
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .unsqueeze(0)
            .to(device)
        )

        labels = (
            batch["labels"]
            .unsqueeze(0)
            .to(device)
        )


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()


        if (step + 1) % 5 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Step {step + 1}/{len(tokenized)} "
                f"| Loss: {loss.item():.4f}"
            )


    average_loss = (
        total_loss / len(tokenized)
    )

    print(
        f"Epoch {epoch + 1} complete "
        f"| Average loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# 11. Save adapter
# --------------------------------------------------

print()
print("Saving LoRA adapter...")

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Adapter saved to: {OUTPUT_PATH}")
print("Training complete!")