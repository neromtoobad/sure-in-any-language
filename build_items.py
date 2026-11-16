"""Build the parallel item set from Belebele (facebook/belebele, CC-BY-SA 4.0).

Every item exists in all five languages with the same passage, question and options
(human-translated), so any difference between languages is the model's, not the question's.

Writes data/items.jsonl. Items are in a fixed shuffled order (seed 0): `run_eval.py --n N`
takes the first N item ids, so a run can be extended later without changing the earlier items.
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANGS = {"en": "eng_Latn", "sw": "swh_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}

by_lang = {}
for lang, code in LANGS.items():
    rows = [json.loads(l) for l in open(ROOT / f"data/{code}.jsonl", encoding="utf-8") if l.strip()]
    by_lang[lang] = {(r["link"], int(r["question_number"])): r for r in rows}

keys = sorted(set.intersection(*(set(d) for d in by_lang.values())))
random.Random(0).shuffle(keys)

with open(ROOT / "data/items.jsonl", "w", encoding="utf-8") as f:
    for rank, k in enumerate(keys):
        gold = {int(by_lang[l][k]["correct_answer_num"]) for l in LANGS}
        assert len(gold) == 1, k  # same gold option in every language
        for lang in LANGS:
            r = by_lang[lang][k]
            f.write(json.dumps({
                "uid": f"q{rank:03d}", "rank": rank, "lang": lang,
                "passage": r["flores_passage"].strip(), "question": r["question"].strip(),
                "options": [r[f"mc_answer{i}"].strip() for i in (1, 2, 3, 4)],
                "gold": int(r["correct_answer_num"]),  # 1..4, index into options
            }, ensure_ascii=False) + "\n")

print(f"{len(keys)} parallel items x {len(LANGS)} languages -> data/items.jsonl")
