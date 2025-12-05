# ============================
# Past Meets Present – HTTP LLM Evaluation (Directory Version)
# - Loops over all JSON files in DATASET_DIR
# - Uses ALL questions in each file in original order
# - Saves per-question and summary results as CSV + Excel
# - Skips any JSON file if BOTH its CSV and XLSX outputs already exist
# ============================

import json
import re
from pathlib import Path

import requests
import pandas as pd

# ========= 1. CONFIG =========

# Folder containing your question_*.json files
DATASET_DIR = Path("question_datasets")

# Base directory for results
RESULTS_BASE_DIR = Path("results")

# HTTP LLM endpoint + model id (OpenAI-compatible API)
API_URL = "http://192.168.0.218:80/v1/chat/completions"  # <-- change to your endpoint
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"           # <-- change to your model name

# High-level system prompt
SYSTEM_PROMPT = (
    "You are an expert historical reasoning model. "
    "Your only job is to choose the correct option."
)

MAX_NEW_TOKENS = 128  # kept for compatibility if you later want to pass it through
N_SHOW = 5            # how many example Q&As to print per file

# Make model name filesystem-safe and create results dir
MODEL_SAFE = MODEL_ID.replace("/", "_")
RESULTS_DIR = RESULTS_BASE_DIR / MODEL_SAFE
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ========= 2. Remote LLM helper =========

def call_llm(
    api_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
) -> str:
    """
    Call an OpenAI-compatible chat/completions HTTP endpoint and return the text output.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        # If your server supports it, you could add e.g. "max_tokens": MAX_NEW_TOKENS
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and data["choices"]:
            c = data["choices"][0]
            # Chat-style response
            if "message" in c and "content" in c["message"]:
                return c["message"]["content"].strip()
            # Text-completion fallback
            if "text" in c:
                return c["text"].strip()
            return json.dumps(c)

        return json.dumps(data)

    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def ask_model(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Thin wrapper to mirror the old signature.
    """
    return call_llm(
        api_url=API_URL,
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
    )


# ========= 3. Helper functions =========

def extract_tf_answer(text: str) -> str:
    """Return 'True', 'False', or 'Unknown' from model text."""
    t = text.strip().lower()
    if t.startswith("true"):
        return "True"
    if t.startswith("false"):
        return "False"
    if "true" in t and "false" not in t:
        return "True"
    if "false" in t and "true" not in t:
        return "False"
    return "Unknown"


def extract_mcq_answer(text: str) -> str:
    """Return 'A'/'B'/'C'/'D' or 'Unknown'."""
    t = text.strip()
    m = re.search(r"\b([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(option|choice)\s*([A-D])", t, re.IGNORECASE)
    if m:
        return m.group(2).upper()
    return "Unknown"


def build_prompt_tf(q_obj):
    return (
        "You are a historian. Answer the following question ONLY with the single word "
        "'True' or 'False'. Do not add any explanation.\n\n"
        f"Question: {q_obj['question']}\n\nAnswer:"
    )


def build_prompt_mcq(q_obj):
    # options keys are 'a','b','c','d' (lowercase). Show as uppercase.
    opts = "\n".join(f"{k.upper()}. {v}" for k, v in q_obj["options"].items())
    return (
        "You are a historian. Choose the correct option and answer ONLY with the "
        "option letter (A, B, C, or D). Do not add any explanation.\n\n"
        f"Question: {q_obj['question']}\n{opts}\n\nAnswer:"
    )


def determine_format(q_obj):
    """Map your 'type' field to 'TF' or 'MCQ'."""
    t = q_obj.get("type", "").strip().upper()
    if "T" in t:   # e.g., 'T/F'
        return "TF"
    return "MCQ"


# ========= 4. Evaluate a single JSON file =========

def evaluate_dataset_file(dataset_path: Path):
    print("\n======================================")
    print(f"[INFO] Evaluating file: {dataset_path.name}")
    print("======================================")

    # Output paths for this file
    csv_path = RESULTS_DIR / f"{dataset_path.stem}.csv"
    xlsx_path = RESULTS_DIR / f"{dataset_path.stem}.xlsx"

    # ---- Skip logic: if BOTH CSV and XLSX already exist, skip this file ----
    if csv_path.exists() and xlsx_path.exists():
        print(f"[SKIP] Both outputs already exist for {dataset_path.name}:")
        print(f"       {csv_path}")
        print(f"       {xlsx_path}")
        return

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at {dataset_path}")
        return

    # Optional diagnostic
    print(f"[INFO] Checking content of {dataset_path}")
    try:
        with dataset_path.open('r', encoding='utf-8') as f:
            content = f.read()
            print(f"[DEBUG] File content (first 300 chars):\n{content[:300]}")
            if not content.strip():
                print("[ERROR] File is empty or contains only whitespace.")
                return
    except Exception as e:
        print(f"[ERROR] Could not read file content: {e}")
        return

    # Load dataset
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        print("[ERROR] Dataset JSON must be a list of question objects.")
        return

    print(f"[INFO] Loaded {len(dataset)} questions from {dataset_path}")

    # Sample structural check
    required_fields = {"question_id", "question", "type", "options", "answer", "template_type"}
    missing_fields = set()
    for q in dataset[:5]:
        for rf in required_fields:
            if rf not in q:
                missing_fields.add(rf)

    if missing_fields:
        print(f"[WARN] Some required fields are missing in sample questions: {missing_fields}")
    else:
        print("[INFO] Sample questions look structurally OK for this script.")

    # Use ALL data as test_data (no shuffle)
    test_data = dataset
    total_questions = len(test_data)
    print(f"[INFO] Evaluating on ALL {total_questions} questions in original order.")

    # ===== Evaluate =====
    overall_correct = 0
    overall_total = 0
    per_template_stats = {}
    per_format_stats = {}
    results_rows = []
    examples_shown = 0

    for idx, q in enumerate(test_data, start=1):
        qid = q.get("question_id")
        # ---- Progress print per question ----
        print(f"[{idx}/{total_questions}] Evaluating question_id={qid} ...")

        q_type = q.get("template_type", "unknown")
        fmt = determine_format(q)           # "TF" or "MCQ"
        question_text = q["question"]
        gold_letter = q["answer"].lower()   # 'a','b','c','d'

        # Build prompt
        if fmt == "TF":
            prompt = build_prompt_tf(q)
        else:
            prompt = build_prompt_mcq(q)

        raw_answer = ask_model(prompt)

        # Parse model answer into a letter matching your schema
        if fmt == "TF":
            tf_pred = extract_tf_answer(raw_answer)  # 'True'/'False'/Unknown
            if tf_pred == "True":
                pred_letter = "a"   # assuming options['a'] = True, ['b'] = False
            elif tf_pred == "False":
                pred_letter = "b"
            else:
                pred_letter = "unknown"
        else:
            mcq_pred = extract_mcq_answer(raw_answer)  # 'A'..'D' or 'Unknown'
            pred_letter = mcq_pred.lower() if mcq_pred != "Unknown" else "unknown"

        is_correct = int(pred_letter == gold_letter)
        overall_correct += is_correct
        overall_total += 1

        # per-template stats
        if q_type not in per_template_stats:
            per_template_stats[q_type] = {"correct": 0, "total": 0}
        per_template_stats[q_type]["correct"] += is_correct
        per_template_stats[q_type]["total"] += 1

        # per-format stats
        if fmt not in per_format_stats:
            per_format_stats[fmt] = {"correct": 0, "total": 0}
        per_format_stats[fmt]["correct"] += is_correct
        per_format_stats[fmt]["total"] += 1

        # save row for CSV/Excel
        results_rows.append({
            "question_id": qid,
            "template_type": q_type,
            "question_format": fmt,
            "question": question_text,
            "gold_answer_letter": gold_letter,
            "gold_answer_text": q["options"][gold_letter],
            "model_raw_answer": raw_answer,
            "model_parsed_letter": pred_letter,
            "model_parsed_text": q["options"].get(pred_letter, "") if pred_letter in q["options"] else "",
            "is_correct": is_correct,
        })

        # show a few examples (first N in the original sequence)
        if examples_shown < N_SHOW:
            print("\n==============================")
            print(f"QID {qid} | Type: {q_type} | Format: {fmt}")
            print("Q:", question_text)
            for k, v in q["options"].items():
                print(f"  {k.upper()}. {v}")
            print("MODEL RAW:", raw_answer)
            print("PRED LETTER:", pred_letter, "| GOLD LETTER:", gold_letter, "| CORRECT?", bool(is_correct))
            examples_shown += 1

    # ===== Summary =====
    overall_acc = overall_correct / max(overall_total, 1)
    print("\n==============================")
    print(f"[RESULT] {dataset_path.name} – Overall accuracy: {overall_acc:.3f}")
    print(f"[RESULT] Total questions evaluated: {overall_total}")

    print("\n[RESULT] Accuracy per template_type:")
    for t, stats in per_template_stats.items():
        acc = stats["correct"] / max(stats["total"], 1)
        total_q = stats["total"]
        print(f"  - {t}: {acc:.3f}  (n={total_q})")

    print("\n[RESULT] Accuracy per question_format:")
    for fmt, stats in per_format_stats.items():
        acc = stats["correct"] / max(stats["total"], 1)
        total_q = stats["total"]
        print(f"  - {fmt}: {acc:.3f}  (n={total_q})")

    # ===== Save to CSV & Excel =====
    results_df = pd.DataFrame(results_rows)

    # ---- Build summary rows for Excel ----
    summary_rows = []

    # Overall
    summary_rows.append({
        "metric": "overall_accuracy",
        "group": "ALL",
        "value": overall_acc,
        "n": overall_total,
    })

    # Per template_type
    for t, stats in per_template_stats.items():
        acc = stats["correct"] / max(stats["total"], 1)
        summary_rows.append({
            "metric": "accuracy_per_template_type",
            "group": t,
            "value": acc,
            "n": stats["total"],
        })

    # Per question_format
    for fmt, stats in per_format_stats.items():
        acc = stats["correct"] / max(stats["total"], 1)
        summary_rows.append({
            "metric": "accuracy_per_question_format",
            "group": fmt,
            "value": acc,
            "n": stats["total"],
        })

    summary_df = pd.DataFrame(summary_rows)

    # ---- Write CSV (per-question only) ----
    results_df.to_csv(csv_path, index=False)

    # ---- Write Excel with two sheets ----
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="per_question")
        summary_df.to_excel(writer, index=False, sheet_name="summary")

    print(f"\n[INFO] Saved per-question results to CSV: {csv_path}")
    print(f"[INFO] Saved per-question + summary results to Excel: {xlsx_path}")
    print("Open the Excel file to see sheets: 'per_question' and 'summary'.")


# ========= 5. MAIN: loop over the directory =========

def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] DATASET_DIR does not exist: {DATASET_DIR}")
        return

    json_files = sorted(DATASET_DIR.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {DATASET_DIR}")
        return

    print(f"[INFO] Found {len(json_files)} JSON files in {DATASET_DIR}")
    print(f"[INFO] Results will be stored in: {RESULTS_DIR}")

    for path in json_files:
        evaluate_dataset_file(path)


if __name__ == "__main__":
    main()