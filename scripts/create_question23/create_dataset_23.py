import json
import random
import re
import pandas as pd

random.seed(42)  # optional, reproducible

# ----------------- Config -----------------

# How close in years events must be to be used in questions.
# Smaller = harder (e.g., 5), larger = easier (e.g., 15 or 20).
CLOSE_WINDOW = 5

# ----------------- Helper cleaning functions -----------------


def clean_name(name: str):
    """
    Clean an event name:
      - strip whitespace
      - drop if empty or contains 'unknown'
    """
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    if "unknown" in name.lower():
        return None
    return name


def safe_year(value):
    """Safely convert a year value to int; return None if invalid."""
    try:
        return int(str(value).strip())
    except Exception:
        return None


YEAR_PATTERN = re.compile(r"\b\d{4}\b")


def sanitize_for_question(name: str) -> str:
    """
    Prepare event name for question text:
      - use clean_name
      - remove standalone 4-digit year tokens (e.g. '1995')
      - collapse extra spaces
    """
    name = clean_name(name)
    if name is None:
        return ""
    # Remove explicit 4-digit years so the model can't just read the year
    name = YEAR_PATTERN.sub("", name)
    # Collapse whitespace
    name = " ".join(name.split())
    return name


# ----------------- Load and prepare events -----------------

df = pd.read_csv("hf://datasets/infinite-dataset-hub/TechAdvancements/data.csv")

# Keep only Event and Year
df_stripped_filtered = df[["Event", "Year"]]

# Turn into a cleaned list of dicts
events = []
for _, row in df_stripped_filtered.iterrows():
    name = clean_name(row["Event"])
    if name is None:
        continue
    year = safe_year(row["Year"])
    if year is None:
        continue
    events.append({
        "Event": name,
        "Year": year
    })

# If there aren't many events, this will naturally cap the number of questions
n_events = len(events)
if n_events < 5:
    raise ValueError(f"Not enough clean events to build questions: {n_events}")

# ----------------- Build 250 True/False questions (harder) -----------------
# "The X happened before the Y."
# Only use pairs where the years are within CLOSE_WINDOW of each other.

tf_questions = []
max_tf = 250

# We'll sample random unordered pairs without repeating the same pair
used_pairs = set()

# Max possible unordered pairs
max_pairs = (n_events * (n_events - 1)) // 2

while len(tf_questions) < max_tf and len(used_pairs) < max_pairs:
    i, j = random.sample(range(n_events), 2)
    pair_key = tuple(sorted((i, j)))
    if pair_key in used_pairs:
        continue

    e1 = events[i]
    e2 = events[j]
    y1 = e1["Year"]
    y2 = e2["Year"]

    # Only keep events that are close in time to increase difficulty
    if abs(y1 - y2) > CLOSE_WINDOW:
        continue

    used_pairs.add(pair_key)

    q_text = (
        f"The {sanitize_for_question(e1['Event'])} happened before "
        f"the {sanitize_for_question(e2['Event'])}."
    )

    truth = y1 < y2

    tf_questions.append({
        "question_id": len(tf_questions) + 1,  # 1–250 (or fewer)
        "question": q_text,
        "type": "T/F",
        "options": {
            "a": "True",
            "b": "False"
        },
        "answer": "a" if truth else "b"
    })

# ----------------- Precompute candidate lists for MCQs (harder) -----------------
# We want:
#   Given event1, which of these events was invented AFTER it? (or BEFORE it)
#   Exactly one correct option.
# All options must be within CLOSE_WINDOW years of event1.

after_candidates = {}       # i -> ( [j indices with Year_j > Year_i & close], [j indices with Year_j <= Year_i & close] )
before_candidates = {}      # i -> ( [j indices with Year_j < Year_i & close], [j indices with Year_j >= Year_i & close] )

for i, e1 in enumerate(events):
    y1 = e1["Year"]
    after_list = []
    after_not_list = []
    before_list = []
    before_not_list = []

    for j, e2 in enumerate(events):
        if i == j:
            continue
        y2 = e2["Year"]

        # Only consider events close in time to e1
        if abs(y2 - y1) > CLOSE_WINDOW:
            continue

        # AFTER direction
        if y2 > y1:
            after_list.append(j)
        else:
            after_not_list.append(j)

        # BEFORE direction
        if y2 < y1:
            before_list.append(j)
        else:
            before_not_list.append(j)

    # Store only if we will potentially be able to use this event as a stem
    if len(after_list) >= 1 and len(after_not_list) >= 3:
        after_candidates[i] = (after_list, after_not_list)
    if len(before_list) >= 1 and len(before_not_list) >= 3:
        before_candidates[i] = (before_list, before_not_list)

# ----------------- Build up to 250 MCQ questions -----------------

mcq_questions = []
max_mcq = 250
used_combinations = set()
letters = ["a", "b", "c", "d"]

while len(mcq_questions) < max_mcq and (after_candidates or before_candidates):
    # Choose a base event index that has at least one viable direction
    possible_indices = set(after_candidates.keys()) | set(before_candidates.keys())
    if not possible_indices:
        break

    i = random.choice(list(possible_indices))
    e1 = events[i]

    # Decide whether to use "after" or "before" for this question
    directions = []
    if i in after_candidates:
        directions.append("after")
    if i in before_candidates:
        directions.append("before")
    if not directions:
        # no valid direction for this i anymore
        after_candidates.pop(i, None)
        before_candidates.pop(i, None)
        continue

    direction = random.choice(directions)

    if direction == "after":
        correct_pool, wrong_pool = after_candidates[i]
        prompt_word = "after"
    else:  # "before"
        correct_pool, wrong_pool = before_candidates[i]
        prompt_word = "before"

    if len(correct_pool) < 1 or len(wrong_pool) < 3:
        # Can't use this direction anymore; remove and continue
        if direction == "after":
            after_candidates.pop(i, None)
        else:
            before_candidates.pop(i, None)
        continue

    # Exactly one correct + three incorrect
    correct_j = random.choice(correct_pool)
    wrong_js = random.sample(wrong_pool, 3)

    combo_key = (i, direction, correct_j, tuple(sorted(wrong_js)))
    if combo_key in used_combinations:
        continue
    used_combinations.add(combo_key)

    option_indices = [correct_j] + wrong_js
    random.shuffle(option_indices)

    options = {}
    correct_letter = None
    for letter, j in zip(letters, option_indices):
        e2 = events[j]
        options[letter] = sanitize_for_question(e2["Event"])
        if j == correct_j:
            correct_letter = letter

    question_text = (
        f"Given the {sanitize_for_question(e1['Event'])}, which of the following events "
        f"was invented {prompt_word} it?"
    )

    mcq_questions.append({
        "question_id": len(tf_questions) + len(mcq_questions) + 1,
        "question": question_text,
        "type": "MCQ",
        "options": options,
        "answer": correct_letter
    })

# ----------------- Save combined questions -----------------

all_questions = tf_questions + mcq_questions

with open("questions23_mixed3.json", "w") as f:
    json.dump(all_questions, f, indent=2)
