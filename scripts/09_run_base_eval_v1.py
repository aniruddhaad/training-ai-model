import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Paths
# ============================================================

BASE_MODEL = r"C:\FinMod\models\SmolLM2-360M"

EVAL_FILE = r"C:\FinMod\data\finance_eval_v1.jsonl"

OUTPUT_DIR = r"C:\FinMod\outputs"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "finance_eval_v1_base_results.jsonl"
)


# ============================================================
# Generation settings
# ============================================================

MAX_NEW_TOKENS = 60

# Deterministic generation
DO_SAMPLE = False


# ============================================================
# Stop markers
# ============================================================

STOP_MARKERS = [
    "### Instruction:",
    "### Question:",
    "### Response:",
    "Instruction:",
    "Question:",
    "Response:",
]


# ============================================================
# Load evaluation dataset
# ============================================================

def load_eval_dataset(path):

    examples = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


# ============================================================
# Clean generated answer
# ============================================================

def clean_answer(answer):

    answer = answer.strip()

    for marker in STOP_MARKERS:

        if marker in answer:

            answer = answer.split(marker)[0].strip()

    return answer


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Finance Evaluation v1 - BASE MODEL")
    print("=" * 70)

    print(f"Base model : {BASE_MODEL}")
    print(f"Eval file  : {EVAL_FILE}")
    print(f"Output     : {OUTPUT_FILE}")
    print()

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------

    examples = load_eval_dataset(EVAL_FILE)

    print(f"Loaded {len(examples)} evaluation examples.")

    if len(examples) != 25:

        print(
            f"WARNING: Expected 25 examples, "
            f"but found {len(examples)}."
        )

    print()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Load BASE model only
    # --------------------------------------------------------

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto"
    )

    model.eval()

    print("Base model loaded.")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model.to(device)

    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    results = []

    print("=" * 70)
    print("Running BASE MODEL evaluation")
    print("=" * 70)

    for index, example in enumerate(examples, start=1):

        example_id = example["id"]

        category = example["category"]

        question = example["question"]

        expected_answer = example["expected_answer"]

        # ----------------------------------------------------
        # Same prompt format as the LoRA evaluation
        # ----------------------------------------------------

        prompt = (
            "### Instruction:\n"
            f"{question}\n\n"
            "### Response:\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Deterministic generation
        # ----------------------------------------------------

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # ----------------------------------------------------
        # Decode only newly generated tokens
        # ----------------------------------------------------

        input_length = inputs["input_ids"].shape[1]

        generated_ids = output_ids[0][input_length:]

        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        answer = clean_answer(answer)

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "id": example_id,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": answer,
        }

        results.append(result)

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        print()
        print("-" * 70)
        print(f"[{index}/{len(examples)}] {example_id}")
        print(f"Category : {category}")
        print(f"Question : {question}")
        print()
        print("MODEL ANSWER:")
        print(answer)
        print()
        print("EXPECTED:")
        print(expected_answer)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
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

    print()
    print("=" * 70)
    print("BASE MODEL evaluation complete")
    print("=" * 70)

    print("Results saved to:")
    print(OUTPUT_FILE)

    print()
    print(f"Total examples: {len(results)}")


if __name__ == "__main__":
    main()