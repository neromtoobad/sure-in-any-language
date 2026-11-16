# Sure in any language

Do small AI models still know when they are wrong once the question is in Hausa, Igbo, Swahili or Yoruba?

This repository holds the code, raw model outputs and report for an entry to the
Apart Research AI x Epistemics Research Sprint (November 13–15, 2026), Track 1: Model Epistemics and Decision-Making Evals.

The report is [`report/report.pdf`](report/report.pdf). All numbers in it are in
[`results/tables.md`](results/tables.md) and [`results/summary.json`](results/summary.json).

## Result in brief

Accuracy / mean confidence on the same 300 questions, and how well confidence separates right
answers from wrong ones (AUROC; 0.5 means not at all):

| Model (4-bit) | English | Swahili | Hausa | Igbo | Yoruba | AUROC, English → Yoruba |
|---|---|---|---|---|---|---|
| Tiny Aya Global 3.35B | 82% / 88% | 75% / 82% | 53% / 65% | 42% / 62% | 43% / 64% | 0.83 → 0.63 |
| Gemma 3 4B | 84% / 99% | 62% / 99% | 44% / 98% | 35% / 98% | 35% / 98% | 0.79 → 0.57 |
| Llama 3.2 3B | 82% / 81% | 53% / 55% | 35% / 43% | 32% / 40% | 32% / 41% | 0.84 → 0.53 |
| Phi-4-mini 3.8B | 89% / 94% | 62% / 76% | 40% / 55% | 39% / 53% | 35% / 51% | 0.87 → 0.59 |

Accuracy falls in every model outside English. Average confidence stays put in one model and falls in
others, but in none of them does it still pick out the right answers in Igbo or Yoruba. A confidence
cut-off tuned on English to allow 10% wrong answers lets through several times that. Checking an
answer against the model's own answer to the English version of the question works much better.
The full write-up is in the report.

## What it measures

The same 300 reading-comprehension questions from [Belebele](https://huggingface.co/datasets/facebook/belebele)
are put to each model in English, Swahili, Hausa, Igbo and Yoruba. The questions are translated by people
and are identical across languages, so a difference between languages comes from the model. For every
question and language the runner records:

| Pass | What is recorded |
|---|---|
| `orig` | probability of each option letter A–D as the first token; the answer and the model's confidence in it |
| `rev` | the same with the four options in reverse order (does the answer survive?) |
| `verbal` | the number the model gives when asked how likely its answer is to be correct |
| `ptrue` | probability of "Yes" when asked whether its answer is correct |

The analysis then asks how far confidence tracks accuracy in each language, whether a confidence
cut-off tuned on English still holds elsewhere, and whether two checks that need no labels (answer
stability under reversed options, agreement with the model's own English answer) do better.

Models, all run locally with llama.cpp as 4-bit GGUF files: Tiny Aya Global 3.35B, Gemma 3 4B,
Llama 3.2 3B, Phi-4-mini 3.8B. Tiny Aya Global is also run at 8, 3 and 2 bits.

## Reproduce

```bash
uv venv && uv pip install numpy matplotlib          # analysis only needs these two
# 1. data: the five Belebele files
for l in eng_Latn swh_Latn hau_Latn ibo_Latn yor_Latn; do
  curl -L -o data/$l.jsonl https://huggingface.co/datasets/facebook/belebele/resolve/main/data/$l.jsonl
done
python build_items.py                               # -> data/items.jsonl (900 questions x 5 languages)
# 2. runs: needs a llama.cpp `llama-server` binary (set LLAMA_SERVER) and the GGUF files
python run_eval.py path/to/model.gguf my-run --n 300
# 3. numbers, tables, figures
python analyze.py                                   # -> results/, report/fig/
# 4. report
report/build_pdf.sh report                          # -> report/report.pdf (needs Google Chrome)
```

Step 3 works straight from this repository: the raw outputs of every run are in `runs/*/answers.jsonl`
(one line per question, language and pass, with the full A–D probabilities), so every number can be
recomputed without running a model. `report/fill.py` puts those numbers into the report text; none is
typed by hand.

## Files

| Path | What |
|---|---|
| `build_items.py` | builds the parallel question set from Belebele |
| `run_eval.py` | runs one GGUF model over the questions through `llama-server`; resumable |
| `queue.sh` | the order the models were run in |
| `analyze.py` | metrics, bootstrap intervals, tables and figures |
| `diag_tokens.py` | diagnostic: which first tokens a model puts probability on besides A–D |
| `runs/<model>/answers.jsonl`, `meta.json` | raw outputs, prompts and chat template of each run |
| `results/` | `summary.json` (every number) and `tables.md` |
| `report/` | report source (`body.html`, `report.html`), `fill.py`, `build_pdf.sh`, `pages.py`, figures, PDF |

## What was built during the sprint, and what was not

Everything in this repository was written and run during the sprint: the code, all model runs, the
analysis and the report.

Reused from outside this project: a llama.cpp build (b12403) and the Tiny Aya GGUF files (Cohere's
official 4-bit files of Tiny Aya Global and Tiny Aya Earth, the official 16-bit file, and 3-bit and
2-bit files quantized from it). These were already on the same machine for a separate safety project
by the author. The 8-bit file was made from the 16-bit file for this study. No code or results from
that project are used here.

## Use of AI tools

The code, the reference check and the draft of the report were produced with an AI assistant
(Claude, through Claude Code). The author chose the project, ran the experiments locally and reviewed
the results and text. See the LLM Usage Statement in the report.

## Data licence

Belebele is CC-BY-SA 4.0 (Meta). It is not copied into this repository; `runs/` holds only question
ids, model probabilities and the gold option number.
