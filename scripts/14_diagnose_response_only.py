"""
14_diagnose_response_only.py

Verify whether Transformers' DataCollatorForLanguageModeling (mlm=False)
preserves manually-created response-only labels.

Run from C:\FinMod:
    python scripts\14_diagnose_response_only.py
"""

from transformers import AutoTokenizer, DataCollatorForLanguageModeling


MODEL_PATH = r"C:\FinMod\models\SmolLM2-360M"


def main():
    print("=" * 72)
    print("RESPONSE-ONLY LABEL MASKING DIAGNOSTIC")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # SmolLM2 does not define a padding token. For a causal LM, using EOS
    # as padding is a standard approach and does not add a vocabulary token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Question: What is an asset?\n\nAnswer:"
    answer = " An asset is a resource controlled by a company."

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)

    # Intended response-only labels:
    # prompt = -100, answer = actual token IDs
    labels = ([-100] * len(prompt_ids)) + answer_ids

    example = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    print(f"\nPadding token: {tokenizer.pad_token!r}")
    print(f"EOS token:     {tokenizer.eos_token!r}")

    print("\nBEFORE collator")
    print("-" * 72)
    print("Prompt tokens:", len(prompt_ids))
    print("Answer tokens:", len(answer_ids))
    print("Total tokens: ", len(input_ids))
    print("Masked (-100):", labels.count(-100))
    print("Trainable labels:", sum(x != -100 for x in labels))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    batch = collator([example])
    after_labels = batch["labels"][0].tolist()

    prompt_masked_after = all(
        x == -100 for x in after_labels[:len(prompt_ids)]
    )
    answer_unmasked_after = all(
        x != -100 for x in after_labels[len(prompt_ids):]
    )

    print("\nAFTER DataCollatorForLanguageModeling")
    print("-" * 72)
    print("Masked (-100):", after_labels.count(-100))
    print("Trainable labels:", sum(x != -100 for x in after_labels))
    print("Prompt still masked:    ", prompt_masked_after)
    print("Answer still trainable: ", answer_unmasked_after)

    print("\nVERDICT")
    print("-" * 72)

    if prompt_masked_after and answer_unmasked_after:
        print("PASS: The collator preserved response-only labels.")
        print("The masking approach is valid.")
    else:
        print("FAIL: The collator overwrote the manually-created labels.")
        print("The previous response-only training method must be fixed.")
        print("Do NOT start the large accounting training run yet.")

    print("\nLabel preview:")
    print("  prompt:", after_labels[:len(prompt_ids)])
    print("  answer:", after_labels[len(prompt_ids):])


if __name__ == "__main__":
    main()
