import json
import random
import unicodedata
import re
from itertools import permutations

# ----------------- Config -----------------

# How close in years events must be to be used in questions
# Smaller = harder (e.g., 3), larger = easier (e.g., 10)
CLOSE_WINDOW = 5

# For compound T/F questions, allow a slightly wider window
COMPOUND_WINDOW = CLOSE_WINDOW * 3

# Max questions of each type
MAX_TF = 250            # total T/F (compound + simple + negation style)
MAX_TF_COMPOUND = 150   # cap on 3-event compound T/Fs
MAX_TF_NEGATION = 100    # cap on "no earlier than / no later than" T/Fs
MAX_MCQ = 250

random.seed(42)  # reproducible

# ----------------- Helpers -----------------


def sanitize_text(s):
    """
    Clean text for JSON output:
      - cast to str
      - strip whitespace
      - normalize unicode
      - replace double quotes with single quotes
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace('"', "'")
    s = unicodedata.normalize("NFKC", s)
    return s


def get_year(event):
    """Safely get the year as an int from an event dict."""
    return int(str(event["Year"]).strip())


YEAR_PATTERN = re.compile(r"\b\d{4}\b")


def clean_events(raw_events):
    """
    Clean a list of events:
      - sanitize name
      - drop events whose name contains 'unknown'
      - drop events whose name contains a 4-digit year (to avoid giving hints)
      - ensure Year is a valid int
    Returns a new list of events with Name of Incident (str) and Year (int).
    """
    cleaned = []
    for e in raw_events:
        name = sanitize_text(e.get("Name of Incident", ""))
        if not name:
            continue
        # Drop anything with 'unknown' in the name (corrupted / placeholder)
        if "unknown" in name.lower():
            continue
        # Drop events that explicitly mention a year in their name
        if YEAR_PATTERN.search(name):
            continue
        try:
            year = get_year(e)
        except Exception:
            continue
        cleaned.append({
            "Name of Incident": name,
            "Year": year,
        })
    return cleaned


def name_for_question(event_dict):
    """Return the incident name for use in question text."""
    return sanitize_text(event_dict["Name of Incident"])


# ----------------- Load & clean events from JSON -----------------

with open("event1.json", "r", encoding="utf-8") as read_file:
    event1s_raw = json.load(read_file)

with open("event2.json", "r", encoding="utf-8") as file:
    event2s_raw = json.load(file)

# Clean both lists to prevent 'Unknown' corruption / bad years / explicit years in names
event1s = clean_events(event1s_raw)
event2s = clean_events(event2s_raw)

# Merge into a single event list and deduplicate by (Name, Year)
all_raw_events = event1s + event2s
seen = set()
events = []
for e in all_raw_events:
    key = (e["Name of Incident"], e["Year"])
    if key in seen:
        continue
    seen.add(key)
    events.append(e)

n_events = len(events)
if n_events < 5:
    raise ValueError(f"Not enough clean events to build questions: {n_events}")

# ----------------- Build compound T/F questions -----------------
# Form:
#   "The A occurred before B and after C."
# Some are true, some false.
# All three years are kept reasonably close.

tf_questions = []
used_triples = set()
tries = 0
MAX_TRIES_COMPOUND = 5000  # safety cap

while len(tf_questions) < min(MAX_TF_COMPOUND, MAX_TF) and tries < MAX_TRIES_COMPOUND:
    tries += 1
    i, j, k = random.sample(range(n_events), 3)
    triple_key = tuple(sorted((i, j, k)))
    if triple_key in used_triples:
        continue

    eA = events[i]
    eB = events[j]
    eC = events[k]
    yA = eA["Year"]
    yB = eB["Year"]
    yC = eC["Year"]

    years = [yA, yB, yC]
    if max(years) - min(years) > COMPOUND_WINDOW:
        continue

    used_triples.add(triple_key)

    # Sort by year to identify earliest, middle, latest
    idxs = [i, j, k]
    idxs_sorted = sorted(idxs, key=lambda idx: events[idx]["Year"])
    earliest_idx, middle_idx, latest_idx = idxs_sorted

    make_true = random.choice([True, False])

    if make_true:
        # True statement: middle before latest AND after earliest
        A_idx = middle_idx
        B_idx = latest_idx
        C_idx = earliest_idx
        truth = True
    else:
        # False statement:
        # Use earliest as A, latest as B, middle as C:
        # A before B is true, but A after C is false -> overall false
        A_idx = earliest_idx
        B_idx = latest_idx
        C_idx = middle_idx
        truth = False

    A = events[A_idx]
    B = events[B_idx]
    C = events[C_idx]

    q_text = (
        f"The {name_for_question(A)} occurred before the {name_for_question(B)} "
        f"and after the {name_for_question(C)}."
    )

    tf_questions.append({
        "question_id": len(tf_questions) + 1,
        "question": q_text,
        "type": "T/F",
        "options": {"a": "True", "b": "False"},
        "answer": "a" if truth else "b"
    })

# ----------------- Extra T/F: "no later than / no earlier than" -----------------
# Form:
#   "The X occurred no later than Y and no earlier than Z."
# True iff Year(Z) <= Year(X) <= Year(Y).

tries2 = 0
MAX_TRIES_NEGATION = 5000

while (
    len(tf_questions) < min(MAX_TF, MAX_TF_COMPOUND + MAX_TF_NEGATION)
    and tries2 < MAX_TRIES_NEGATION
):
    tries2 += 1
    i, j, k = random.sample(range(n_events), 3)
    eX = events[i]
    eY = events[j]
    eZ = events[k]
    yX = eX["Year"]
    yY = eY["Year"]
    yZ = eZ["Year"]

    years = [yX, yY, yZ]
    if max(years) - min(years) > COMPOUND_WINDOW:
        continue

    # Decide randomly if we want the statement to be true or false
    should_be_true = random.choice([True, False])
    cond_true = (yZ <= yX <= yY)

    # Only accept if reality matches our requested truth value
    if should_be_true != cond_true:
        continue

    q_text = (
        f"The {name_for_question(eX)} occurred no later than the {name_for_question(eY)} "
        f"and no earlier than the {name_for_question(eZ)}."
    )

    tf_questions.append({
        "question_id": len(tf_questions) + 1,
        "question": q_text,
        "type": "T/F",
        "options": {"a": "True", "b": "False"},
        "answer": "a" if should_be_true else "b"
    })

# ----------------- Build simple pairwise T/F questions -----------------
# Use *pairs* of events whose years are close.
# Randomize "before" vs "after" wording.

tf_candidates = []

for i, e1 in enumerate(events):
    y1 = e1["Year"]
    for j, e2 in enumerate(events):
        if i == j:
            continue
        y2 = e2["Year"]

        # Only keep pairs that are close in time to increase difficulty
        if abs(y1 - y2) <= CLOSE_WINDOW:
            tf_candidates.append((i, j))

random.shuffle(tf_candidates)

for (idx1, idx2) in tf_candidates:
    if len(tf_questions) >= MAX_TF:
        break

    e1 = events[idx1]
    e2 = events[idx2]
    y1 = e1["Year"]
    y2 = e2["Year"]

    if random.choice(["before", "after"]) == "before":
        q_text = (
            f"The {name_for_question(e1)} occurred before "
            f"the {name_for_question(e2)}."
        )
        truth = y1 < y2
    else:
        q_text = (
            f"The {name_for_question(e1)} occurred after "
            f"the {name_for_question(e2)}."
        )
        truth = y1 > y2

    tf_questions.append({
        "question_id": len(tf_questions) + 1,
        "question": q_text,
        "type": "T/F",
        "options": {"a": "True", "b": "False"},
        "answer": "a" if truth else "b"
    })

# ----------------- Precompute candidate lists for MCQs -----------------
# MCQ styles:
#   1) "Which event occurred BEFORE X?"                 (before)
#   2) "Which event occurred AFTER X?"                  (after)
#   3) "Which event occurred CLOSEST IN TIME after X?" (closest_after)
#   4) Ordering: "Which option lists these events in chronological order?" (order)
#   5) NEW: "Which statement about X, Y, Z is correct?" (relational)

before_candidates = {}        # i -> (earlier_list, not_earlier_list)
after_candidates = {}         # i -> (later_list, not_later_list)
closest_after_candidates = {} # i -> later_list (need >= 4)
order_candidates = {}         # i -> neighbors list (events within CLOSE_WINDOW)
relational_candidates = {}    # i -> neighbors list (for 3-event statement MCQs)

for i, e1 in enumerate(events):
    y1 = e1["Year"]
    neighbors = []
    earlier_list = []
    not_earlier_list = []
    later_list = []
    not_later_list = []

    for j, e2 in enumerate(events):
        if i == j:
            continue
        y2 = e2["Year"]

        # Only consider events that are close in time to e1
        if abs(y2 - y1) > CLOSE_WINDOW:
            continue

        neighbors.append(j)

        if y2 < y1:
            earlier_list.append(j)
            not_later_list.append(j)  # also not later
        elif y2 > y1:
            later_list.append(j)
            not_earlier_list.append(j)  # also not earlier
        else:
            # same year: neither strictly earlier nor later
            not_earlier_list.append(j)
            not_later_list.append(j)

    # BEFORE-style MCQ
    if len(earlier_list) >= 1 and len(not_earlier_list) >= 3:
        before_candidates[i] = (earlier_list, not_earlier_list)

    # AFTER-style MCQ
    if len(later_list) >= 1 and len(not_later_list) >= 3:
        after_candidates[i] = (later_list, not_later_list)

    # CLOSEST-AFTER-style MCQ (need at least 4 later events)
    if len(later_list) >= 4:
        closest_after_candidates[i] = later_list

    # ORDER-style MCQ (use 3 neighbors + base = 4 total events)
    if len(neighbors) >= 3:
        order_candidates[i] = neighbors

    # RELATIONAL-style MCQ (need at least 2 neighbors to form X,Y,Z)
    if len(neighbors) >= 2:
        relational_candidates[i] = neighbors

# ----------------- Build MCQ questions -----------------

mcq_questions = []
used_combinations = set()
letters = ["a", "b", "c", "d"]

while len(mcq_questions) < MAX_MCQ and (
    before_candidates
    or after_candidates
    or closest_after_candidates
    or order_candidates
    or relational_candidates
):
    possible_indices = (
        set(before_candidates.keys())
        | set(after_candidates.keys())
        | set(closest_after_candidates.keys())
        | set(order_candidates.keys())
        | set(relational_candidates.keys())
    )
    if not possible_indices:
        break

    i = random.choice(list(possible_indices))
    e1 = events[i]
    y1 = e1["Year"]

    # Decide which MCQ style is available for this base event
    directions = []
    if i in before_candidates:
        directions.append("before")
    if i in after_candidates:
        directions.append("after")
    if i in closest_after_candidates:
        directions.append("closest_after")
    if i in order_candidates:
        directions.append("order")
    if i in relational_candidates:
        directions.append("relational")

    if not directions:
        before_candidates.pop(i, None)
        after_candidates.pop(i, None)
        closest_after_candidates.pop(i, None)
        order_candidates.pop(i, None)
        relational_candidates.pop(i, None)
        continue

    direction = random.choice(directions)

    if direction == "before":
        correct_pool, wrong_pool = before_candidates[i]
        if len(correct_pool) < 1 or len(wrong_pool) < 3:
            before_candidates.pop(i, None)
            continue

        correct_j = random.choice(correct_pool)
        wrong_js = random.sample(wrong_pool, 3)

        combo_key = (i, "before", correct_j, tuple(sorted(wrong_js)))
        if combo_key in used_combinations:
            continue
        used_combinations.add(combo_key)

        option_indices = [correct_j] + wrong_js
        random.shuffle(option_indices)

        options = {}
        correct_letter = None
        for letter, j in zip(letters, option_indices):
            e2 = events[j]
            options[letter] = name_for_question(e2)
            if j == correct_j:
                correct_letter = letter

        question_text = (
            f"Given the {name_for_question(e1)}, which of the following events "
            f"occurred before it?"
        )

    elif direction == "after":
        correct_pool, wrong_pool = after_candidates[i]
        if len(correct_pool) < 1 or len(wrong_pool) < 3:
            after_candidates.pop(i, None)
            continue

        correct_j = random.choice(correct_pool)
        wrong_js = random.sample(wrong_pool, 3)

        combo_key = (i, "after", correct_j, tuple(sorted(wrong_js)))
        if combo_key in used_combinations:
            continue
        used_combinations.add(combo_key)

        option_indices = [correct_j] + wrong_js
        random.shuffle(option_indices)

        options = {}
        correct_letter = None
        for letter, j in zip(letters, option_indices):
            e2 = events[j]
            options[letter] = name_for_question(e2)
            if j == correct_j:
                correct_letter = letter

        question_text = (
            f"Given the {name_for_question(e1)}, which of the following events "
            f"occurred after it?"
        )

    elif direction == "closest_after":
        later_list = closest_after_candidates[i]
        if len(later_list) < 4:
            closest_after_candidates.pop(i, None)
            continue

        # Choose 4 later events; correct is the closest in time after e1
        option_indices = random.sample(later_list, 4)
        option_indices_sorted = sorted(
            option_indices,
            key=lambda j: events[j]["Year"] - y1
        )
        correct_j = option_indices_sorted[0]

        combo_key = (i, "closest_after", tuple(sorted(option_indices)))
        if combo_key in used_combinations:
            continue
        used_combinations.add(combo_key)

        random.shuffle(option_indices)

        options = {}
        correct_letter = None
        for letter, j in zip(letters, option_indices):
            e2 = events[j]
            options[letter] = name_for_question(e2)
            if j == correct_j:
                correct_letter = letter

        question_text = (
            f"Given the {name_for_question(e1)}, which of the following events "
            f"occurred closest in time after it?"
        )

    elif direction == "order":
        neighbors = order_candidates[i]
        if len(neighbors) < 3:
            order_candidates.pop(i, None)
            continue

        # Choose 3 neighbors + base = 4 events
        chosen_neighbors = random.sample(neighbors, 3)
        idxs = [i] + chosen_neighbors

        # True chronological order by year
        idxs_sorted = sorted(idxs, key=lambda idx: events[idx]["Year"])
        correct_order = idxs_sorted

        all_perms = list(permutations(idxs))
        all_perms = [p for p in all_perms if list(p) != correct_order]
        if len(all_perms) < 3:
            order_candidates.pop(i, None)
            continue

        wrong_orders = random.sample(all_perms, 3)

        option_seqs = [correct_order] + list(wrong_orders)

        combo_key = (tuple(sorted(idxs)), "order")
        if combo_key in used_combinations:
            continue
        used_combinations.add(combo_key)

        random.shuffle(option_seqs)

        options = {}
        correct_letter = None

        def seq_to_str(seq):
            names = [name_for_question(events[idx]) for idx in seq]
            return " → ".join(names)

        for letter, seq in zip(letters, option_seqs):
            seq_str = seq_to_str(seq)
            options[letter] = seq_str
            if list(seq) == correct_order:
                correct_letter = letter

        set_names = ", ".join(sorted(
            {name_for_question(events[idx]) for idx in idxs}
        ))
        question_text = (
            f"Consider the following events: {set_names}. "
            f"Which option lists them in the correct chronological order, "
            f"from earliest to latest?"
        )

    else:  # direction == "relational"
        neighbors = relational_candidates[i]
        if len(neighbors) < 2:
            relational_candidates.pop(i, None)
            continue

        # Pick two distinct neighbors to form a triple X (base), Y, Z
        j, k = random.sample(neighbors, 2)
        eX = e1
        eY = events[j]
        eZ = events[k]
        yX = eX["Year"]
        yY = eY["Year"]
        yZ = eZ["Year"]

        # Keep triple reasonably close in time
        triple_years = [yX, yY, yZ]
        if max(triple_years) - min(triple_years) > COMPOUND_WINDOW:
            continue

        # Define the 4 candidate statements
        statements = [
            (
                f"The {name_for_question(eX)} occurred before the {name_for_question(eY)} "
                f"and after the {name_for_question(eZ)}.",
                (yX < yY and yX > yZ),
            ),
            (
                f"The {name_for_question(eX)} occurred after the {name_for_question(eY)} "
                f"and before the {name_for_question(eZ)}.",
                (yX > yY and yX < yZ),
            ),
            (
                f"The {name_for_question(eX)} occurred before both the {name_for_question(eY)} "
                f"and the {name_for_question(eZ)}.",
                (yX < yY and yX < yZ),
            ),
            (
                f"The {name_for_question(eX)} occurred after both the {name_for_question(eY)} "
                f"and the {name_for_question(eZ)}.",
                (yX > yY and yX > yZ),
            ),
        ]

        true_indices = [idx for idx, (_, cond) in enumerate(statements) if cond]
        # Only use if exactly one statement is true
        if len(true_indices) != 1:
            continue

        true_idx = true_indices[0]

        combo_key = (tuple(sorted([i, j, k])), "relational")
        if combo_key in used_combinations:
            continue
        used_combinations.add(combo_key)

        options = {}
        correct_letter = None
        for letter, idx_stmt in zip(letters, range(len(statements))):
            text, cond = statements[idx_stmt]
            options[letter] = text
            if idx_stmt == true_idx:
                correct_letter = letter

        set_names = ", ".join(sorted({
            name_for_question(eX),
            name_for_question(eY),
            name_for_question(eZ),
        }))
        question_text = (
            f"Consider the following events: {set_names}. "
            f"Which of the following statements about their chronological order is correct?"
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

with open("questions21_mixed_hard4.json", "w", encoding="utf-8") as file:
    json.dump(all_questions, file, indent=2, ensure_ascii=False)
