import json
import random
from itertools import permutations

# ----------------- Config -----------------

# How close in years two countries must be to be used in a question.
# Smaller = harder (e.g., 3 or 5), larger = easier (e.g., 10 or 20).
CLOSE_WINDOW = 5

# For compound T/F questions, allow a slightly wider window
COMPOUND_WINDOW = CLOSE_WINDOW * 3

# Max questions of each type
MAX_TF = 250                # total T/F (compound + negation + pairwise)
MAX_TF_COMPOUND = 150       # cap on 3-country compound T/Fs
MAX_TF_NEGATION = 100        # cap on "no earlier than / no later than" T/Fs
MAX_MCQ = 250

random.seed(42)  # reproducible

# ----------------- Helpers -----------------


def clean_name(name):
    """
    Clean a country name:
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
    """Safely convert established year to int; return None if invalid."""
    try:
        return int(str(value).strip())
    except Exception:
        return None


def name_for_question(country_dict):
    """Return the country name for use in question text."""
    return country_dict["country"]


# ----------------- Load & clean countries -----------------

with open("countries.json", "r") as read_file:
    countries_raw = json.load(read_file)

countries = []
for c in countries_raw:
    name = clean_name(c.get("country"))
    if name is None:
        continue

    year = safe_year(c.get("established"))
    if year is None:
        continue

    countries.append({
        "country": name,
        "established": year,
    })

# Shuffle to break any ordering from the original file
random.shuffle(countries)

n_countries = len(countries)
if n_countries < 5:
    raise ValueError(f"Not enough clean countries to build questions: {n_countries}")

# ----------------- Build compound T/F questions -----------------
# Form:
#   "A was established before B and after C."
# Some are true, some false.
# All three years are kept reasonably close.

tf_questions = []
used_triples = set()
tries = 0
MAX_TRIES_COMPOUND = 5000  # safety cap

while len(tf_questions) < min(MAX_TF_COMPOUND, MAX_TF) and tries < MAX_TRIES_COMPOUND:
    tries += 1
    i, j, k = random.sample(range(n_countries), 3)
    triple_key = tuple(sorted((i, j, k)))
    if triple_key in used_triples:
        continue

    cA = countries[i]
    cB = countries[j]
    cC = countries[k]
    yA = cA["established"]
    yB = cB["established"]
    yC = cC["established"]

    years = [yA, yB, yC]
    if max(years) - min(years) > COMPOUND_WINDOW:
        continue

    used_triples.add(triple_key)

    # Sort by year to identify earliest, middle, latest
    idxs = [i, j, k]
    idxs_sorted = sorted(idxs, key=lambda idx: countries[idx]["established"])
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

    A = countries[A_idx]
    B = countries[B_idx]
    C = countries[C_idx]

    q_text = (
        f"{name_for_question(A)} was established before {name_for_question(B)} "
        f"and after {name_for_question(C)}."
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
#   "X was established no later than Y and no earlier than Z."
# True iff Year(Z) <= Year(X) <= Year(Y)

tries2 = 0
MAX_TRIES_NEGATION = 5000

while (
    len(tf_questions) < min(MAX_TF, MAX_TF_COMPOUND + MAX_TF_NEGATION)
    and tries2 < MAX_TRIES_NEGATION
):
    tries2 += 1
    i, j, k = random.sample(range(n_countries), 3)
    cX = countries[i]
    cY = countries[j]
    cZ = countries[k]
    yX = cX["established"]
    yY = cY["established"]
    yZ = cZ["established"]

    years = [yX, yY, yZ]
    if max(years) - min(years) > COMPOUND_WINDOW:
        continue

    should_be_true = random.choice([True, False])
    cond_true = (yZ <= yX <= yY)

    if should_be_true != cond_true:
        continue

    q_text = (
        f"{name_for_question(cX)} was established no later than {name_for_question(cY)} "
        f"and no earlier than {name_for_question(cZ)}."
    )

    tf_questions.append({
        "question_id": len(tf_questions) + 1,
        "question": q_text,
        "type": "T/F",
        "options": {"a": "True", "b": "False"},
        "answer": "a" if should_be_true else "b"
    })

# ----------------- Build simple pairwise T/F questions -----------------
# Use *pairs* of countries whose establishment years are close.
# Randomize "before" vs "after" wording.

tf_candidates = []

for i, c1 in enumerate(countries):
    y1 = c1["established"]
    for j, c2 in enumerate(countries):
        if i == j:
            continue
        y2 = c2["established"]

        # Only keep pairs that are close in time to increase difficulty
        if abs(y1 - y2) <= CLOSE_WINDOW:
            tf_candidates.append((i, j))

random.shuffle(tf_candidates)

for (idx1, idx2) in tf_candidates:
    if len(tf_questions) >= MAX_TF:
        break

    c1 = countries[idx1]
    c2 = countries[idx2]
    y1 = c1["established"]
    y2 = c2["established"]

    # Randomly choose phrasing: "before" or "after"
    if random.choice(["before", "after"]) == "before":
        q_text = f"{name_for_question(c1)} was established before {name_for_question(c2)}."
        truth = y1 < y2
    else:
        q_text = f"{name_for_question(c1)} was established after {name_for_question(c2)}."
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
#   1) "Which country was established BEFORE X?"                 (before)
#   2) "Which country was established AFTER X?"                  (after)
#   3) "Which country was established CLOSEST IN TIME after X?" (closest_after)
#   4) Ordering: "Which option lists these countries in chronological order?" (order)
#   5) NEW: "Which statement about X, Y, Z is correct?"         (relational)

before_candidates = {}        # i -> (earlier_list, not_earlier_list)
after_candidates = {}         # i -> (later_list, not_later_list)
closest_after_candidates = {} # i -> later_list (need >= 4)
order_candidates = {}         # i -> neighbors list (countries within CLOSE_WINDOW)
relational_candidates = {}    # i -> neighbors list (for 3-country statement MCQs)

for i, c1 in enumerate(countries):
    y1 = c1["established"]
    neighbors = []
    earlier_list = []
    not_earlier_list = []
    later_list = []
    not_later_list = []

    for j, c2 in enumerate(countries):
        if i == j:
            continue
        y2 = c2["established"]

        # Only consider countries that are close in time to c1
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

    # CLOSEST-AFTER-style MCQ (need at least 4 later countries)
    if len(later_list) >= 4:
        closest_after_candidates[i] = later_list

    # ORDER-style MCQ (use 3 neighbors + base = 4 total countries)
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
    c1 = countries[i]
    y1 = c1["established"]

    # Decide which MCQ style is available for this base country
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
            c2 = countries[j]
            options[letter] = name_for_question(c2)
            if j == correct_j:
                correct_letter = letter

        question_text = (
            f"Given {name_for_question(c1)}, which of the following countries "
            f"was established before it?"
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
            c2 = countries[j]
            options[letter] = name_for_question(c2)
            if j == correct_j:
                correct_letter = letter

        question_text = (
            f"Given {name_for_question(c1)}, which of the following countries "
            f"was established after it?"
        )

    elif direction == "closest_after":
        later_list = closest_after_candidates[i]
        if len(later_list) < 4:
            closest_after_candidates.pop(i, None)
            continue

        # Choose 4 later countries; correct is the closest in time after c1
        option_indices = random.sample(later_list, 4)
        option_indices_sorted = sorted(
            option_indices,
            key=lambda j: countries[j]["established"] - y1
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
            c2 = countries[j]
            options[letter] = name_for_question(c2)
            if j == correct_j:
                correct_letter = letter

        question_text = (
            f"Given {name_for_question(c1)}, which of the following countries "
            f"was established closest in time after it?"
        )

    elif direction == "order":
        neighbors = order_candidates[i]
        if len(neighbors) < 3:
            order_candidates.pop(i, None)
            continue

        # Choose 3 neighbors + base = 4 countries
        chosen_neighbors = random.sample(neighbors, 3)
        idxs = [i] + chosen_neighbors

        # Sort by year to get true chronological order
        idxs_sorted = sorted(idxs, key=lambda idx: countries[idx]["established"])
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
            names = [name_for_question(countries[idx]) for idx in seq]
            return " -> ".join(names)

        for letter, seq in zip(letters, option_seqs):
            seq_str = seq_to_str(seq)
            options[letter] = seq_str
            if list(seq) == correct_order:
                correct_letter = letter

        set_names = ", ".join(sorted(
            {name_for_question(countries[idx]) for idx in idxs}
        ))
        question_text = (
            f"Consider the following countries: {set_names}. "
            f"Which option lists them in the correct chronological order of establishment, "
            f"from earliest to latest?"
        )

    else:  # direction == "relational"
        neighbors = relational_candidates[i]
        if len(neighbors) < 2:
            relational_candidates.pop(i, None)
            continue

        # Pick two distinct neighbors to form a triple X (base), Y, Z
        j, k = random.sample(neighbors, 2)
        cX = c1
        cY = countries[j]
        cZ = countries[k]
        yX = cX["established"]
        yY = cY["established"]
        yZ = cZ["established"]

        triple_years = [yX, yY, yZ]
        if max(triple_years) - min(triple_years) > COMPOUND_WINDOW:
            continue

        # Four candidate statements
        statements = [
            (
                f"{name_for_question(cX)} was established before {name_for_question(cY)} "
                f"and after {name_for_question(cZ)}.",
                (yX < yY and yX > yZ),
            ),
            (
                f"{name_for_question(cX)} was established after {name_for_question(cY)} "
                f"and before {name_for_question(cZ)}.",
                (yX > yY and yX < yZ),
            ),
            (
                f"{name_for_question(cX)} was established before both "
                f"{name_for_question(cY)} and {name_for_question(cZ)}.",
                (yX < yY and yX < yZ),
            ),
            (
                f"{name_for_question(cX)} was established after both "
                f"{name_for_question(cY)} and {name_for_question(cZ)}.",
                (yX > yY and yX > yZ),
            ),
        ]

        true_indices = [idx for idx, (_, cond) in enumerate(statements) if cond]
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
            name_for_question(cX),
            name_for_question(cY),
            name_for_question(cZ),
        }))
        question_text = (
            f"Consider the following countries: {set_names}. "
            f"Which of the following statements about their chronological order "
            f"of establishment is correct?"
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

with open("questions22_mixed_hard2.json", "w") as write_file:
    json.dump(all_questions, write_file, indent=4)
