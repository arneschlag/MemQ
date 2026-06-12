import os
from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your_key"),
    base_url="https://api.deepseek.com"
)

TYPE1_TEMPLATE = """Act as a SPARQL expert. 
I need you to explain the meaning and function of a specific part of a SPARQL query.
You job is answer the Question for me. ONLY OUTPUT THE ANSWER, NOTING ELSE!!

### EXAMPLE1
Sparql:
?entity1 ns:location.country.currency_used ?entity2 .
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the currency used in the country ?entity1.

### EXAMPLE2
Sparql:
?entity2 ns:location.country.currency_used ?entity1 .
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the country that use ?entity1 as currency.

### EXAMPLE3
Sparql:
?entity2 ns:government.election_campaign.candidate ?entity1 .
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the election campaign which ?entity1 is the candidate.

### EXAMPLE4
Sparql:
?entity1 ns:government.election_campaign.candidate ?entity2 .
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the candidate in the election campaign ?entity1.

### EXAMPLE5
Sparql:
{{ ?entity2 ns:sports.sports_championship_event.runner_up ?entity1 }} UNION {{ ?entity2 ns:sports.sports_championship_event.champion ?entity1 }}
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is either the runner-up or the champion of a sports championship event ?entity1.

### EXAMPLE6
Sparql:
{{ ?entity1 ns:location.statistical_region.places_exported_to ?tmp0 . ?tmp0 ns:location.imports_and_exports.exported_to ?entity2 }} UNION {{ ?entity1 ns:location.statistical_region.places_exported_from ?tmp1 . ?tmp1 ns:location.imports_and_exports.exported_from ?entity2 }}
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the place that is either exported to or exported from the statistical region ?entity1.

### YOUR TURN
Sparql:
{sparql}
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: """

TYPE2_TEMPLATE = """Act as a SPARQL expert.
I need you to explain the meaning and function of a specific part of a SPARQL query.
You job is answer the Question for me. ONLY OUTPUT THE ANSWER, NOTING ELSE!!

### EXAMPLE1
Sparql:
?cvt ns:government.government_position_held.office_holder ?entity1 .
?entity2 ns:government.governmental_body.members ?cvt . 
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the governmental body that has an office holder ?entity1.

### EXAMPLE2 
Sparql:
?entity1 ns:film.actor.film ?cvt .
?cvt ns:film.performance.character ?entity2 .
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the character played by the actor ?entity1.

### EXAMPLE3 
Sparql:
?cvt ns:music.group_membership.member ?entity1 .
?entity2 ns:music.musical_group.member ?cvt .
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity2 is the musical group that has the member ?entity1.

### YOUR TURN
Sparql:
{sparql}
Question: How does ?entity2 related to ?entity1 ? Please answer the question with "?entity2 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer:  
"""

TYPE3_TEMPLATE = """Act as a SPARQL expert. 
I need you to explain the meaning and function of a specific part of a SPARQL query. 
You job is complete the answer for me. ONLY OUTPUT THE ANSWER, NOTING ELSE!! 

### EXAMPLE1 
Sparql: 
?cvt ns:sports.sports_team_coach_tenure.position ?entity1 . 
?cvt ns:sports.sports_team_coach_tenure.coach ?entity2 . 
?entity3 ns:sports.sports_team.coaches ?cvt . 
Question: How does ?entity3 related to ?entity1 and ?entity2 ? Please answer the question with "?entity3 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity3 is the sports team that has a coach ?entity2 who holds the position ?entity1 . 

### EXAMLPE2 
Sparql: 
?entity1 ns:film.actor.film ?cvt . 
?cvt ns:film.performance.character ?entity2 . 
?cvt ns:film.performance.film ?entity3 . 
Question: How does ?entity3 related to ?entity1 and ?entity2 ? Please answer the question with "?entity3 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity3 is the film in which the actor ?entity1 performs the character ?entity2. 

### EXAMLPE3 
Sparql: 
?entity1 ns:sports.pro_athlete.teams ?cvt . 
?cvt ns:sports.sports_team_roster.team ?entity2 . 
?cvt ns:sports.sports_team_roster.from ?entity3 
Question: How does ?entity3 related to ?entity1 and ?entity2 ? Please answer the question with "?entity3 is [noun phrase]" . Make sure ?cvt not in your answer.
Answer: ?entity3 is the starting date when the professional athlete ?entity1 was part of the team ?entity2. 

### YOUR TURN 
Sparql: 
{sparql} 
Question: How does ?entity3 related to ?entity1 and ?entity2 ? Please answer the question with "?entity3 is [noun phrase]" . Make sure ?cvt not in your answer. 
Answer: 
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MAX_WORKERS = 20  # parallel requests
SAVE_EVERY = 500  # save intermediate results every N keys
RETRIES = 3

_write_lock = threading.Lock()


def get_response(prompt):
    for attempt in range(RETRIES):
        try:
            completion = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=prompt,
                top_p=0.7,
                temperature=0.9,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            import time
            time.sleep(2 ** attempt)


def _explain_one(pair):
    key, messages = pair
    return key, get_response(messages)


def _save_snapshot(data, path):
    with _write_lock:
        with open(path, "w") as f:
            json.dump(data, f)


def load_snapshot(path):
    """Load existing results from a snapshot file, or return empty dict."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def process_keys(keys, template, desc, snapshot_path):
    """Process a list of keys in parallel with progress bar, incremental saves, and resume."""
    results = load_snapshot(snapshot_path)

    # Split template at {sparql}: system = cached prefix, user = key + question
    parts = template.split("{sparql}")
    system_msg = parts[0].rstrip()
    user_suffix = parts[1].lstrip()

    remaining = [
        (k, [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{k}\n{user_suffix}"},
        ])
        for k in keys if k not in results
    ]

    if not remaining:
        print(f"  {desc}: all {len(keys)} already cached — skipping")
        return results

    skipped = len(keys) - len(remaining)
    print(f"  {desc}: {len(remaining)} new / {skipped} cached / {len(keys)} total")

    completed = len(results)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_explain_one, p): p[0] for p in remaining}
        with tqdm(total=len(remaining), desc=desc, initial=0) as pbar:
            for future in as_completed(futures):
                try:
                    key, explanation = future.result()
                    results[key] = explanation
                except Exception as e:
                    key = futures[future]
                    results[key] = f"ERROR: {e}"
                    tqdm.write(f"Failed {key[:60]}... : {e}")

                completed += 1
                pbar.update(1)

                if completed % SAVE_EVERY == 0:
                    _save_snapshot(results, snapshot_path)

    _save_snapshot(results, snapshot_path)
    return results


if __name__ == "__main__":
    with open("output/all_key.json", "r") as f:
        all_key = json.load(f)

    key_explain = {}

    for type_key in ("1", "2", "3"):
        keys = all_key.get(type_key, [])
        if not keys:
            continue

        if type_key == "1":
            tmpl = TYPE1_TEMPLATE
        elif type_key == "2":
            tmpl = TYPE2_TEMPLATE
        else:
            tmpl = TYPE3_TEMPLATE

        print(f"\nType {type_key}: {len(keys)} keys, {MAX_WORKERS} workers")
        result = process_keys(keys, tmpl, f"Type {type_key}", f"output/key_explain{type_key}.json")
        key_explain.update(result)

    with open("output/key_explain.json", "w") as f:
        json.dump(key_explain, f)

    print(f"\nDone: {len(key_explain)} keys explained")