"""Fill report/body.html from results/summary.json so no number in the report is typed by hand.

usage: python report/fill.py            -> writes report/_body_filled.html, lists unknown tokens
Tokens in body.html:
  {{run.lang.metric}}      one formatted number, e.g. {{aya-q4.yo.acc}} -> 43%
  {{run.lang.metric_ci}}   its interval,         e.g. {{aya-q4.yo.gap_ci}} -> [+15, +26]
  {{run.n}}                questions that run answered in all five languages
  {{T:name}}               a generated table (see TABLES below)
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE.parent / "results/summary.json"))
LANGS = ["en", "sw", "ha", "ig", "yo"]
LNAME = {"en": "English", "sw": "Swahili", "ha": "Hausa", "ig": "Igbo", "yo": "Yoruba"}
MODELS = {"aya-q4": "Tiny Aya Global", "gemma3-4b-q4": "Gemma 3 4B", "llama32-3b-q4": "Llama 3.2 3B",
          "phi4-mini-q4": "Phi-4-mini"}
QUANT = {"aya-q8": "8-bit", "aya-q4": "4-bit", "aya-q3": "3-bit", "aya-q2": "2-bit"}
PP = {"gap", "gap_vs_en", "flip_vs_en", "verbal_gap", "gap_T_own", "gap_T_en", "gap_m80", "rawgap_m80"}                       # signed percentage points
NUM = {"ece", "auroc", "brier", "ece_T_own", "ece_T_en", "T_own", "auroc_T_own", "verbal_auroc", "pyes_auroc", "stable_auroc",
       "agree_auroc", "combo_auroc", "tau_en"}                              # plain two-decimal numbers
INT = {"n_sure", "cf_en_n", "cf_own_n", "verbal_mode", "n_m80"}

main = [m for m in MODELS if m in S]
quant = [q for q in QUANT if q in S]


def fmt(metric, v):
    base = metric[:-3] if metric.endswith("_ci") else metric
    if v is None:
        return "–"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(base, x) for x in v) + "]"
    if base in INT:
        return str(int(v))
    if base in NUM:
        return f"{v:.2f}"
    if base in PP:
        r = round(100 * v)
        return "0" if r == 0 else f"{r:+d}".replace("-", "−")
    return f"{100 * v:.0f}%"


def val(run, lang, metric):
    return fmt(metric, S[run]["langs"][lang].get(metric))


def table(head, rows, cls="data"):
    h = "".join(f"<th>{c}</th>" for c in head)
    body = ""
    for r in rows:
        if r[0] == "grp":
            body += f'<tr class="grp"><td colspan="{len(head)}">{r[1]}</td></tr>'
        else:
            body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table class="{cls}"><tr>{h}</tr>{body}</table>'


def matrix(runs, names, lines, langs=LANGS, first=""):
    """One block of rows per run, under a row with its name; each line is (label, function(run, lang) -> text)."""
    rows = []
    for r in runs:
        rows.append(("grp", names[r]))
        for label, f in lines:
            rows.append([label] + [f(r, l) for l in langs])
    return table([first] + [LNAME[l] for l in langs], rows)


def t_main():
    return matrix(main, MODELS, [
        ("Accuracy / mean confidence", lambda r, l: f"{val(r, l, 'acc')} / {val(r, l, 'conf')}"),
        ("Gap, pp [95% CI]", lambda r, l: f"{val(r, l, 'gap')} {val(r, l, 'gap_ci')}"),
        ("AUROC [95% CI]", lambda r, l: f"{val(r, l, 'auroc')} {val(r, l, 'auroc_ci')}"),
    ])


def t_cut():
    def err(r, l, k):
        return "–" if S[r]["langs"][l].get(k) is None else f"{val(r, l, k)} {val(r, l, k + '_ci')}"
    return matrix(main, MODELS, [
        ("English cut-off: answered (number)", lambda r, l: f"{val(r, l, 'cf_en_cov')} ({val(r, l, 'cf_en_n')})"),
        ("&nbsp;&nbsp;&nbsp;of which wrong [95% CI]", lambda r, l: err(r, l, "cf_en_err")),
        ("Own cut-off: answered (number)", lambda r, l: f"{val(r, l, 'cf_own_cov')} ({val(r, l, 'cf_own_n')})"),
        ("&nbsp;&nbsp;&nbsp;of which wrong [95% CI]", lambda r, l: err(r, l, "cf_own_err")),
        ("ECE: raw / English T / own T", lambda r, l: f"{val(r, l, 'ece')} / {val(r, l, 'ece_T_en')} / {val(r, l, 'ece_T_own')}"),
        ("Gap after own T, pp", lambda r, l: val(r, l, "gap_T_own")),
        ("Own T (fitted temperature)", lambda r, l: val(r, l, "T_own")),
        ("AUROC after own T", lambda r, l: val(r, l, "auroc_T_own")),
    ])


def t_free():
    return matrix(main, MODELS, [
        ("Right when same / when different", lambda r, l: f"{val(r, l, 'acc_agree')} / {val(r, l, 'acc_disagree')}"),
        ("Rule: answered / wrong (confidence)",
         lambda r, l: f"{val(r, l, 'both_cov')} / {val(r, l, 'both_err')} ({val(r, l, 'conf_err_same_cov')})"),
    ], langs=LANGS[1:])


def t_free_more():
    na = lambda f: (lambda r, l: "–" if l == "en" else f(r, l))
    return matrix(main, MODELS, [
        ("Flips when options are reversed", lambda r, l: val(r, l, "flip")),
        ("Right when stable / when it flips", lambda r, l: f"{val(r, l, 'acc_stable')} / {val(r, l, 'acc_flipped')}"),
        ("Same answer as in English", na(lambda r, l: val(r, l, "agree_en"))),
        ("Flips minus English, pp [95% CI]", na(lambda r, l: f"{val(r, l, 'flip_vs_en')} {val(r, l, 'flip_vs_en_ci')}")),
        ("AUROC of agreement [95% CI]", na(lambda r, l: f"{val(r, l, 'agree_auroc')} {val(r, l, 'agree_auroc_ci')}")),
        ("Rule: wrong [95% CI]", na(lambda r, l: f"{val(r, l, 'both_err')} {val(r, l, 'both_err_ci')}")),
        ("Top letter: share / when reversed",
         lambda r, l: f"{S[r]['langs'][l]['fav_letter']}: {val(r, l, 'fav_share')} / {val(r, l, 'fav_share_rev')}"),
    ])


def t_say():
    mode = lambda r, l: f"{val(r, l, 'verbal_mode')} ({val(r, l, 'verbal_mode_share')})"
    return matrix(main, MODELS, [
        ("Most common stated number (share of answers)", mode),
        ("Mean stated confidence", lambda r, l: val(r, l, "verbal_mean")),
        ("AUROC of stated number", lambda r, l: val(r, l, "verbal_auroc")),
        ('Says "Yes, correct": all answers / wrong answers',
         lambda r, l: f"{val(r, l, 'pyes_says_yes')} / {val(r, l, 'pyes_yes_when_wrong')}"),
        ("AUROC of yes/no self-check", lambda r, l: val(r, l, "pyes_auroc")),
    ])


def t_quant():
    names = {q: f"{QUANT[q]} (n={S[q]['n']})" for q in quant}
    return matrix(quant, names, [
        ("Accuracy", lambda r, l: val(r, l, "acc")),
        ("Mean confidence", lambda r, l: val(r, l, "conf")),
        ("Gap, pp [95% CI]", lambda r, l: f"{val(r, l, 'gap')} {val(r, l, 'gap_ci')}"),
        ("AUROC of confidence", lambda r, l: val(r, l, "auroc")),
        ("Answer flips when options are reversed", lambda r, l: val(r, l, "flip")),
    ])


def t_extra():
    if S.get("aya-earth-q4", {}).get("n", 0) < S["aya-q4"]["n"]:
        return "<p><i>This run was still in progress when this copy was built.</i></p>"
    runs = [r for r in ("aya-q4", "aya-earth-q4") if r in S]
    names = {"aya-q4": "Tiny Aya Global", "aya-earth-q4": "Tiny Aya Earth"}
    return matrix(runs, names, [
        ("Accuracy", lambda r, l: val(r, l, "acc")),
        ("Mean confidence", lambda r, l: val(r, l, "conf")),
        ("Gap, pp [95% CI]", lambda r, l: f"{val(r, l, 'gap')} {val(r, l, 'gap_ci')}"),
        ("AUROC of confidence", lambda r, l: val(r, l, "auroc")),
        ("Answer flips when options are reversed", lambda r, l: val(r, l, "flip")),
    ])


def t_detail():
    return matrix(main, MODELS, [
        ("Gap minus English gap, pp [95% CI]", lambda r, l: "–" if l == "en" else f"{val(r, l, 'gap_vs_en')} {val(r, l, 'gap_vs_en_ci')}"),
        ("Answers at ≥90% confidence", lambda r, l: val(r, l, "share_sure")),
        ("&nbsp;&nbsp;&nbsp;of which wrong (number of answers)", lambda r, l: f"{val(r, l, 'wrong_when_sure')} ({val(r, l, 'n_sure')})"),
        ("ECE", lambda r, l: val(r, l, "ece")),
        ("Brier score", lambda r, l: val(r, l, "brier")),
        ("Accuracy with options reversed", lambda r, l: val(r, l, "acc_rev")),
        ("First-token probability on A–D", lambda r, l: val(r, l, "letter_mass")),
        ("Questions with ≥80% of it on A–D", lambda r, l: val(r, l, "share_m80")),
        ("&nbsp;&nbsp;&nbsp;gap on those questions, pp", lambda r, l: val(r, l, "gap_m80")),
        ("&nbsp;&nbsp;&nbsp;gap without renormalising, pp", lambda r, l: val(r, l, "rawgap_m80")),
    ])


TABLES = {"main": t_main, "cut": t_cut, "free": t_free, "free_more": t_free_more, "say": t_say, "quant": t_quant, "extra": t_extra,
          "detail": t_detail}


def span(metric, langs, runs):
    """Lowest and highest value of a metric over the given runs and languages, e.g. 36–43%."""
    langs = {"all": LANGS, "af": LANGS[1:]}.get(langs) or langs.split(",")
    vals = [S[r]["langs"][l].get(metric) for r in runs for l in langs]
    vals = [v for v in vals if v is not None]
    lo, hi = fmt(metric, min(vals)), fmt(metric, max(vals))
    if lo == hi:
        return lo
    if metric in PP:
        return f"{lo} to {hi}"
    return f"{lo.rstrip('%')}–{hi}"


def resolve(tok):
    if tok.startswith("T:"):
        return TABLES[tok[2:]]()
    if tok.startswith("R:"):      # {{R:metric:langs}} over the four main models
        _, metric, langs = tok.split(":")
        return span(metric, langs, main)
    if tok.startswith("RQ"):      # {{RQ4:metric:langs}} over languages for one Tiny Aya precision (RQ: all precisions)
        head, metric, langs = tok.split(":")
        return span(metric, langs, quant if head == "RQ" else [f"aya-q{head[2:]}"])
    if tok.startswith("D:"):      # {{D:run.lang.metric:run.lang.metric}} first minus second, in whole points, unsigned
        _, x, y = tok.split(":")
        (r1, l1, m1), (r2, l2, m2) = x.split("."), y.split(".")
        return str(abs(round(100 * S[r1]["langs"][l1][m1]) - round(100 * S[r2]["langs"][l2][m2])))
    if tok == "X:agree_gap":      # largest distance, in whole points, between agreement-with-English rate and accuracy
        return str(max(abs(round(100 * S[r]["langs"][l]["agree_en"]) - round(100 * S[r]["langs"][l]["acc"]))
                       for r in main for l in LANGS[1:]))
    if tok in ("repo", "runs_where", "released"):   # report/repo_url.txt holds the public repository URL, if there is one
        f = HERE / "repo_url.txt"
        u = f.read_text().strip() if f.exists() else ""
        if tok == "repo":
            return (f'Code repository: <a href="{u}">{u.replace("https://", "")}</a>' if u
                    else "Code and raw model outputs: available from the author on request")
        if tok == "runs_where":
            return "in the repository" if u else "kept with the code"
        return "are released" if u else "are available from the author"
    if tok == "Nmodels":
        return {1: "One", 2: "Two", 3: "Three", 4: "Four"}[len(main)]
    if tok == "nmodels":
        return {1: "one", 2: "two", 3: "three", 4: "four"}[len(main)]
    parts = tok.split(".")
    if len(parts) == 2 and parts[1] == "n":
        return str(S[parts[0]]["n"])
    run, lang, metric = parts
    if metric == "fav_letter":
        return S[run]["langs"][lang]["fav_letter"]
    if metric not in S[run]["langs"][lang] and not metric.endswith("_ci"):
        raise KeyError(metric)
    return val(run, lang, metric)


if __name__ == "__main__":
    body = open(HERE / "body.html", encoding="utf-8").read()
    bad = []

    def sub(m):
        try:
            return resolve(m.group(1).strip())
        except Exception as e:
            bad.append(f"{m.group(1)} ({type(e).__name__})")
            return f'<span style="background:#fd0">??{m.group(1)}??</span>'

    out = re.sub(r"\{\{([^{}]+)\}\}", sub, body)
    open(HERE / "_body_filled.html", "w", encoding="utf-8").write(out)
    todo = re.findall(r"[A-Z-]+-TODO", out)
    print(f"filled; runs available: {', '.join(S)}")
    if bad: print("UNKNOWN TOKENS:", ", ".join(sorted(set(bad))))
    if todo: print("still TODO:", ", ".join(sorted(set(todo))))
    sys.exit(1 if bad else 0)
