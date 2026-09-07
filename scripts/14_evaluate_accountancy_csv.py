"""
14_evaluate_accountancy_csv.py

Baseline evaluation runner for the cleaned Class 11 and Class 12 Accountancy
datasets.

Run from C:\FinMod:
    python scripts\14_evaluate_accountancy_csv.py
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(r"C:\FinMod")
MODEL_PATH = BASE_DIR / "models" / "SmolLM2-360M"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

FILES = {
    "Class11": DATA_DIR / "Accountancy_11th_FINAL.csv",
    "Class12": DATA_DIR / "Accountancy_12th_FINAL.csv",
}

EVAL_PER_CLASS = 40
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 180
SEED = 42


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_question(row):
    return clean_text(row.get("Question", ""))


def normalize_answer(row):
    return clean_text(row.get("Answer", ""))


def choose_balanced(rows, n):
    """Select a deterministic, diverse subset across QuestionType and Topic."""
    unique = {}
    for row in rows:
        q = normalize_question(row)
        a = normalize_answer(row)
        if q and a:
            unique[(q.lower(), a.lower())] = row

    rows = list(unique.values())

    groups = defaultdict(list)
    for row in rows:
        qtype = clean_text(row.get("QuestionType", "")) or "Unknown"
        groups[qtype].append(row)

    for group in groups.values():
        group.sort(
            key=lambda r: (
                clean_text(r.get("Topic", "")).lower(),
                normalize_question(r).lower(),
            )
        )

    group_names = sorted(groups)
    pointers = {g: 0 for g in group_names}
    selected = []
    used_questions = set()
    last_topic = None

    while len(selected) < min(n, len(rows)):
        progress = False

        for g in group_names:
            items = groups[g]
            start = pointers[g]

            candidates = [
                i for i in range(start, len(items))
                if clean_text(items[i].get("Topic", "")) != last_topic
            ]
            if not candidates:
                candidates = list(range(start, len(items)))

            if not candidates:
                continue

            i = candidates[0]
            row = items[i]
            pointers[g] = i + 1

            q = normalize_question(row)
            if q.lower() in used_questions:
                continue

            selected.append(row)
            used_questions.add(q.lower())
            last_topic = clean_text(row.get("Topic", ""))
            progress = True

            if len(selected) >= n:
                break

        if not progress:
            break

    return selected


def build_prompt(row):
    return (
        "You are an accounting tutor. Answer the following accountancy "
        "question clearly and accurately.\n\n"
        f"Question: {normalize_question(row)}\n\n"
        "Answer:"
    )


def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    torch.manual_seed(SEED)

    print("=" * 72)
    print("ACCOUNTANCY CSV BASELINE EVALUATION")
    print("=" * 72)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_selected = []

    for class_name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        rows = load_csv(path)
        selected = choose_balanced(rows, EVAL_PER_CLASS)

        print(f"\n{class_name}")
        print(f"  source rows: {len(rows)}")
        print(f"  selected:    {len(selected)}")
        print("  QuestionType distribution:")
        print(Counter(
            clean_text(r.get("QuestionType", "")) or "Unknown"
            for r in selected
        ))

        for i, row in enumerate(selected, start=1):
            all_selected.append({
                "id": f"{class_name}_{i:03d}",
                "class": class_name,
                "topic": clean_text(row.get("Topic", "")),
                "difficulty": clean_text(row.get("Difficulty", "")),
                "student_level": clean_text(row.get("StudentLevel", "")),
                "question_type": clean_text(row.get("QuestionType", "")),
                "question_complexity": clean_text(row.get("QuestionComplexity", "")),
                "prerequisites": clean_text(row.get("Prerequisites", "")),
                "question": normalize_question(row),
                "reference_answer": normalize_answer(row),
            })

    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
    )
    model.eval()

    print(f"Device: {next(model.parameters()).device}")

    output_path = OUTPUT_DIR / "accountancy_eval_v1_base.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(all_selected, start=1):
            prompt = build_prompt(item)
            answer = generate_answer(model, tokenizer, prompt)

            result = {
                **item,
                "prompt": prompt,
                "model_answer": answer,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                f"[{index:03d}/{len(all_selected)}] "
                f"{item['id']} | {item['question'][:75]}"
            )
            print(f"    -> {answer[:180]}")

    report_path = OUTPUT_DIR / "accountancy_eval_v1_base_report.txt"

    class_counts = Counter(x["class"] for x in all_selected)
    qtype_counts = Counter(
        (x["class"], x["question_type"] or "Unknown")
        for x in all_selected
    )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("ACCOUNTANCY EVALUATION V1 - BASE MODEL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Evaluation rows: {len(all_selected)}\n")
        f.write(f"Rows per class target: {EVAL_PER_CLASS}\n\n")

        f.write("Class distribution:\n")
        for k, v in sorted(class_counts.items()):
            f.write(f"  {k}: {v}\n")

        f.write("\nQuestionType distribution:\n")
        for (cls, qtype), count in sorted(qtype_counts.items()):
            f.write(f"  {cls} | {qtype}: {count}\n")

        f.write(
            "\nIMPORTANT:\n"
            "This first pass is a baseline generation set, not an automatic "
            "accuracy score. Reference and model answers need semantic review "
            "before scoring rules are finalized.\n"
        )

    print("\nDONE")
    print(f"Predictions: {output_path}")
    print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()
