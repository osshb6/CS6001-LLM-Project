import json
import random
import pandas as pd
import unicodedata

# ----------------- Helper cleaning functions -----------------

def sanitize_text(s: str) -> str:
    """
    Clean text for JSON output:
      - convert to str
      - strip whitespace
      - normalize unicode
      - replace double quotes with single quotes
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace('"', "'")
    s = unicodedata.normalize("NFKC", s)   # optional but prevents weird chars
    return s

def clean_name(name: str) -> str | None:
    """
    Clean event name:
      - sanitize text
      - skip if empty or contains 'unknown'
    """
    name = sanitize_text(name)
    if not name:
        return None
    if "unknown" in name.lower():
        return None
    return name

def safe_year(value) -> int | None:
    """Safely convert year to int."""
    try:
        return int(str(value).strip())
    except Exception:
        return None

def can_person_live_during_both(year1, year2, max_lifespan=80):
    """Return True if someone could live during both events."""
    return abs(year1 - year2) < max_lifespan

# ----------------- Load and prepare events -----------------

df = pd.read_csv("World Important Dates.csv")

stripped_df = df[["Name of Incident", "Year"]]
df_stripped_filtered = stripped_df[
    ~stripped_df["Year"].str.contains("BC", case=False, na=False)
]

events = []
for _, row in df_stripped_filtered.iterrows():
    name = clean_name(row["Name of Incident"])
    if name is None:
        continue

    year = safe_year(row["Year"])
    if year is None:
        continue

    events.append({
        "Name of Incident": name,
        "Year": year
    })

# Split into two halves
event1s = events[:500]
event2s = events[500:1000]

# ----------------- Build 250 True/False questions -----------------

tf_pairs = list(zip(event1s, event2s))
random.shuffle(tf_pairs)

tf_questions = []

for idx, (e1, e2) in enumerate(tf_pairs):
    if len(tf_questions) >= 250:
        break

    name1 = sanitize_text(e1["Name of Incident"])
    name2 = sanitize_text(e2["Name of Incident"])

    q_text = (
        f"Given the {name1} and the {name2}, it is possible for someone to "
        f"have lived during both of them."
    )

    truth = can_person_live_during_both(e1["Year"], e2["Year"])

    tf_questions.append({
        "question_id": len(tf_questions) + 1,
        "question": q_text,
        "type": "T/F",
        "options": {"a": "True", "b": "False"},
        "answer": "a" if truth else "b"
    })

# ----------------- Precompute valid/invalid for MCQs -----------------

valid_by_event1 = {}
invalid_by_event1 = {}

for i, e1 in enumerate(event1s):
    y1 = e1["Year"]
    valid = []
    invalid = []
    for j, e2 in enumerate(event2s):
        y2 = e2["Year"]
        if can_person_live_during_both(y1, y2):
            valid.append(j)
        else:
            invalid.append(j)

    if len(valid) >= 1 and len(invalid) >= 3:
        valid_by_event1[i] = valid
        invalid_by_event1[i] = invalid

# ----------------- Build 250 MCQ questions -----------------

mcq_questions = []
used_combinations = set()
letters = ["a", "b", "c", "d"]

while len(mcq_questions) < 250 and valid_by_event1:
    e1_index = random.choice(list(valid_by_event1.keys()))
    e1 = event1s[e1_index]

    valid_list = valid_by_event1[e1_index]
    invalid_list = invalid_by_event1[e1_index]

    if len(valid_list) < 1 or len(invalid_list) < 3:
        del valid_by_event1[e1_index]
        del invalid_by_event1[e1_index]
        continue

    correct_j = random.choice(valid_list)
    wrong_js = random.sample(invalid_list, 3)

    combo_key = (e1_index, correct_j, tuple(sorted(wrong_js)))
    if combo_key in used_combinations:
        continue
    used_combinations.add(combo_key)

    option_indices = [correct_j] + wrong_js
    random.shuffle(option_indices)

    options = {}
    correct_letter = None

    for letter, j in zip(letters, option_indices):
        e2 = event2s[j]
        options[letter] = sanitize_text(e2["Name of Incident"])
        if j == correct_j:
            correct_letter = letter

    question_text = (
        f"Given the {sanitize_text(e1['Name of Incident'])}, "
        f"which of the following events would allow for a person to have lived during both incidents?"
    )

    mcq_questions.append({
        "question_id": 250 + len(mcq_questions) + 1,
        "question": question_text,
        "type": "MCQ",
        "options": options,
        "answer": correct_letter
    })

# ----------------- Save Combined -----------------

all_questions = tf_questions + mcq_questions

with open("questions_mixed20.json", "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)
