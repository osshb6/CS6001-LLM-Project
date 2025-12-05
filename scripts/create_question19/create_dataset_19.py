import json
import random

# ------------ Helpers ------------

def can_play(person, game_year):
    """
    Returns True if the person could have played the game:
    birth_year <= game_year <= death_year
    """
    try:
        birth_year = int(person["Birth year"])
        death_year = person["Death year"]
        if not death_year:
            return False
        death_year = int(death_year)
        game_year = int(game_year)
    except (ValueError, TypeError, KeyError):
        return False

    return birth_year <= game_year <= death_year


def sanitize_name(name):
    """
    Sanitize names so JSON doesn't need to escape quotes:
      - cast to string
      - strip whitespace
      - replace " with '
    """
    return str(name).strip().replace('"', "'")


# ------------ Load data ------------

with open("games.json", "r", encoding="utf-8") as f:
    games = json.load(f)

with open("people_and_years.json", "r", encoding="utf-8") as f:
    people = json.load(f)


# ------------ Build 250 True/False questions ------------

tf_questions = []

people_indices = list(range(len(people)))
random.shuffle(people_indices)

for idx in people_indices:
    if len(tf_questions) >= 250:
        break

    person = people[idx]

    # Need a known death year to bound the interval
    if not person.get("Death year"):
        continue

    game = random.choice(games)

    person_name = sanitize_name(person["Name"])
    game_name = sanitize_name(game["name"])

    q_text = f"It is possible that {person_name} played {game_name}."

    truth = can_play(person, game["created"])

    tf_questions.append({
        "question_id": len(tf_questions) + 1,  # 1–250
        "question": q_text,
        "type": "T/F",
        "options": {
            "a": "True",
            "b": "False"
        },
        "answer": "a" if truth else "b"
    })


# ------------ Precompute person lists per game for MCQ ------------

valid_by_game = {}    # game_index -> [person_indices that COULD have played]
invalid_by_game = {}  # game_index -> [person_indices that could NOT have played]

for gi, game in enumerate(games):
    game_year = game.get("created")
    if not game_year:
        continue

    valid = []
    invalid = []
    for pi, person in enumerate(people):
        if not person.get("Death year"):
            continue
        if can_play(person, game_year):
            valid.append(pi)
        else:
            invalid.append(pi)

    # Need at least 1 valid and 3 invalid people to form a 4-option MCQ
    if len(valid) >= 1 and len(invalid) >= 3:
        valid_by_game[gi] = valid
        invalid_by_game[gi] = invalid


# ------------ Build 250 MCQ questions ------------

mcq_questions = []
used_combinations = set()  # to avoid exact duplicate (game + same people set)

letters = ["a", "b", "c", "d"]

while len(mcq_questions) < 250 and valid_by_game:
    gi = random.choice(list(valid_by_game.keys()))
    game = games[gi]

    valid_list = valid_by_game[gi]
    invalid_list = invalid_by_game[gi]

    if len(valid_list) < 1 or len(invalid_list) < 3:
        # Not enough people left for this game; drop it
        del valid_by_game[gi]
        del invalid_by_game[gi]
        continue

    correct_pi = random.choice(valid_list)
    wrong_pis = random.sample(invalid_list, 3)

    combo_key = (gi, correct_pi, tuple(sorted(wrong_pis)))
    if combo_key in used_combinations:
        continue
    used_combinations.add(combo_key)

    # Build options list [correct_person] + [3 wrong_people], then shuffle
    option_person_indices = [correct_pi] + wrong_pis
    random.shuffle(option_person_indices)

    options = {}
    correct_letter = None

    for letter, pi in zip(letters, option_person_indices):
        person_name = sanitize_name(people[pi]["Name"])
        options[letter] = person_name
        if pi == correct_pi:
            correct_letter = letter

    game_name = sanitize_name(game["name"])
    question_text = f"Which of the following people could have played {game_name}?"

    mcq_questions.append({
        "question_id": 250 + len(mcq_questions) + 1,  # 251–500
        "question": question_text,
        "type": "MCQ",
        "options": options,
        "answer": correct_letter
    })


# ------------ Combine and save ------------

all_questions = tf_questions + mcq_questions

with open("questions_mixed.json", "w", encoding="utf-8") as f:
    json.dump(all_questions, f, indent=2, ensure_ascii=False)
