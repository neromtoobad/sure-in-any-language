"""Turn runs/<model>/answers.jsonl into the numbers, tables and figures of the report.

usage: python analyze.py [--runs aya-q4,gemma3-4b-q4,...] [--boot 2000]
Writes results/summary.json, results/tables.md and report/fig/*.png.

Everything is computed on the items a model has answered in all five languages, so every
comparison between languages is paired on the same questions.
"""
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
LANGS = ["en", "sw", "ha", "ig", "yo"]
LNAME = {"en": "English", "sw": "Swahili", "ha": "Hausa", "ig": "Igbo", "yo": "Yoruba"}
MODELS = {  # run name -> display name; order = report order
    "aya-q4": "Tiny Aya Global 3.35B", "gemma3-4b-q4": "Gemma 3 4B", "llama32-3b-q4": "Llama 3.2 3B",
    "phi4-mini-q4": "Phi-4-mini 3.8B",
}
QUANT = {"aya-q8": "8-bit", "aya-q4": "4-bit", "aya-q3": "3-bit", "aya-q2": "2-bit"}
EXTRA = {"aya-earth-q4": "Tiny Aya Earth 3.35B (4-bit)"}

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--alpha", type=float, default=0.10, help="target error rate among answered questions")
a = ap.parse_args()
rng = np.random.default_rng(0)


# ---------- loading ----------
def load(name):
    f = ROOT / "runs" / name / "answers.jsonl"
    if not f.exists():
        return None
    rec = {}
    for l in f.open(encoding="utf-8"):
        r = json.loads(l)
        rec.setdefault((r["uid"], r["lang"]), {})[r["pass"]] = r
    uids = sorted({u for u, _ in rec})
    uids = [u for u in uids if all("orig" in rec.get((u, l), {}) for l in LANGS)]
    if not uids:
        return None
    n = len(uids)

    def arr(pas, key, dtype=float):
        out = np.full((n, len(LANGS)), np.nan)
        for i, u in enumerate(uids):
            for j, l in enumerate(LANGS):
                v = rec[(u, l)].get(pas, {}).get(key)
                if v is not None:
                    out[i, j] = v
        return out

    d = {"uids": uids, "n": n, "gold": arr("orig", "gold"), "pred": arr("orig", "pred"), "conf": arr("orig", "conf"),
         "mass": arr("orig", "letter_mass"), "rev_pred": arr("rev", "pred"), "rev_conf": arr("rev", "conf"),
         "verbal": arr("verbal", "verbal") / 100.0, "p_yes": arr("ptrue", "p_yes")}
    d["probs"] = np.full((n, len(LANGS), 4), np.nan)
    for i, u in enumerate(uids):
        for j, l in enumerate(LANGS):
            d["probs"][i, j] = rec[(u, l)]["orig"]["probs"]
    d["correct"] = (d["pred"] == d["gold"]).astype(float)
    d["verbal_raw"] = {l: Counter(rec[(u, l)]["verbal"]["verbal_raw"] for u in uids if "verbal" in rec[(u, l)]) for l in LANGS}
    return d


# ---------- metrics ----------
def ece(conf, correct, bins=10):
    ok = ~np.isnan(conf); conf, correct = conf[ok], correct[ok]
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return tot


def auroc(score, label):
    """P(score of a correct answer > score of a wrong one), ties count half. nan if undefined."""
    ok = ~np.isnan(score); score, label = score[ok], label[ok]
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    s = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    r = np.empty(len(s)); r[order] = ranks
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ci(samples):
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return lo, hi


def fit_threshold(conf, correct, alpha):
    """Lowest confidence cut-off whose answered questions have error <= alpha (most coverage)."""
    order = np.argsort(-conf)
    c, k = conf[order], correct[order]
    err = 1 - np.cumsum(k) / np.arange(1, len(k) + 1)
    good = np.where(err <= alpha)[0]
    good = [g for g in good if g + 1 == len(c) or c[g + 1] < c[g]]  # only cut between distinct values
    return c[good[-1]] if good else float("inf")


def selective(conf, correct, tau):
    m = conf >= tau
    return (m.mean(), (1 - correct[m]).mean() if m.any() else float("nan"))


def fit_temperature(probs, gold):
    logits = np.log(np.clip(probs, 1e-6, 1))
    best, best_t = 1e9, 1.0
    for t in np.exp(np.linspace(math.log(0.1), math.log(50), 140)):
        z = logits / t; z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(gold)), gold.astype(int) - 1].mean()
        if nll < best:
            best, best_t = nll, t
    return best_t


def apply_temperature(probs, t):
    z = np.log(np.clip(probs, 1e-6, 1)) / t; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p.max(1)


def out_flip_en(d, en):
    return 1 - (d["rev_pred"][:, en] == d["pred"][:, en]).mean()


def model_stats(d):
    n = d["n"]; out = {"n": n, "langs": {}}
    boot = rng.integers(0, n, size=(a.boot, n))
    acc_b = d["correct"][boot].mean(1); conf_b = d["conf"][boot].mean(1)   # (boot, langs)
    gap_b = conf_b - acc_b
    K = 5   # cross-fitting folds: fit on four fifths of the questions, score the remaining fifth
    folds = np.arange(n) % K
    en = LANGS.index("en")
    for j, l in enumerate(LANGS):
        c, k = d["conf"][:, j], d["correct"][:, j]
        s = {"acc": k.mean(), "conf": c.mean(), "gap": c.mean() - k.mean(), "ece": ece(c, k), "auroc": auroc(c, k),
             "brier": ((c - k) ** 2).mean(), "letter_mass": np.nanmean(d["mass"][:, j])}
        s["acc_ci"] = ci(acc_b[:, j]); s["conf_ci"] = ci(conf_b[:, j]); s["gap_ci"] = ci(gap_b[:, j])
        # robustness: only questions where the letters A-D took at least 80% of the first-token probability
        m80 = d["mass"][:, j] >= 0.8
        s["n_m80"] = int(m80.sum()); s["share_m80"] = m80.mean()
        s["acc_m80"] = k[m80].mean() if m80.any() else float("nan")
        s["gap_m80"] = (c[m80].mean() - k[m80].mean()) if m80.any() else float("nan")
        s["rawgap_m80"] = ((c * d["mass"][:, j])[m80].mean() - k[m80].mean()) if m80.any() else float("nan")
        s["gap_vs_en"] = s["gap"] - (d["conf"][:, en].mean() - d["correct"][:, en].mean())
        s["gap_vs_en_ci"] = ci(gap_b[:, j] - gap_b[:, en])
        s["acc_vs_en_ci"] = ci(acc_b[:, j] - acc_b[:, en]); s["conf_vs_en_ci"] = ci(conf_b[:, j] - conf_b[:, en])
        hi = c >= 0.9
        s["share_sure"] = hi.mean(); s["wrong_when_sure"] = (1 - k[hi]).mean() if hi.any() else float("nan")
        s["n_sure"] = int(hi.sum())
        # letter habit: how often the model picks its favourite letter, whatever option sits behind it
        shown = d["pred"][:, j]
        cnt = Counter(shown.astype(int)); fav, nfav = cnt.most_common(1)[0]
        s["fav_letter"] = "ABCD"[fav - 1]; s["fav_share"] = nfav / n
        s["gold_on_fav"] = (d["gold"][:, j] == fav).mean()          # how often that letter is in fact correct
        if (~np.isnan(d["rev_pred"][:, j])).any():
            s["fav_share_rev"] = ((5 - d["rev_pred"][:, j]) == fav).mean()   # same letter when options are reversed
        # stated confidence and yes/no self-check
        v = d["verbal"][:, j]
        if (~np.isnan(v)).any():
            vals = v[~np.isnan(v)]
            s["verbal_mean"] = vals.mean(); s["verbal_gap"] = vals.mean() - k[~np.isnan(v)].mean()
            s["verbal_auroc"] = auroc(v, k); s["verbal_parsed"] = (~np.isnan(v)).mean()
            top = Counter(np.round(vals * 100).astype(int)).most_common(1)[0]
            s["verbal_mode"] = int(top[0]); s["verbal_mode_share"] = top[1] / len(vals)
        py = d["p_yes"][:, j]
        if (~np.isnan(py)).any():
            s["pyes_mean"] = np.nanmean(py); s["pyes_auroc"] = auroc(py, k)
            s["pyes_says_yes"] = np.nanmean(py > 0.5)
            s["pyes_yes_when_wrong"] = np.nanmean(py[k == 0] > 0.5) if (k == 0).any() else float("nan")
        # option-order stability
        rp = d["rev_pred"][:, j]
        if (~np.isnan(rp)).any():
            stable = (rp == d["pred"][:, j]).astype(float)
            s["flip"] = 1 - stable.mean(); s["flip_ci"] = ci(1 - stable[boot].mean(1))
            s["stable_auroc"] = auroc(stable, k)
            s["acc_stable"] = k[stable == 1].mean() if stable.any() else float("nan")
            s["acc_flipped"] = k[stable == 0].mean() if (stable == 0).any() else float("nan")
            s["acc_rev"] = (rp == d["gold"][:, j]).mean()
        # agreement with the model's own English answer
        if l != "en":
            agree = (d["pred"][:, j] == d["pred"][:, en]).astype(float)
            s["agree_en"] = agree.mean(); s["agree_auroc"] = auroc(agree, k)
            s["acc_agree"] = k[agree == 1].mean() if agree.any() else float("nan")
            s["acc_disagree"] = k[agree == 0].mean() if (agree == 0).any() else float("nan")
            if (~np.isnan(rp)).any():
                both = agree * stable
                s["both_cov"] = both.mean(); s["both_err"] = 1 - k[both == 1].mean() if both.any() else float("nan")
                s["combo_auroc"] = auroc(c * (0.5 + 0.5 * agree) * (0.5 + 0.5 * stable), k)
                # same number of answers, chosen by confidence instead: the most confident ones
                kk = int(both.sum()); top = np.argsort(-c, kind="mergesort")[:kk]
                s["conf_err_same_cov"] = 1 - k[top].mean() if kk else float("nan")
            ka = int(agree.sum()); top = np.argsort(-c, kind="mergesort")[:ka]
            s["conf_err_agree_cov"] = 1 - k[top].mean() if ka else float("nan")
            s["agree_err"] = 1 - s["acc_agree"]
            s["acc_en_same_q"] = d["correct"][:, en].mean()   # accuracy of simply answering the English version
        # abstain rule: answer only if confidence >= tau.  tau from English vs tau from the language itself.
        tau_en_all = fit_threshold(d["conf"][:, en], d["correct"][:, en], a.alpha)
        s["tau_en"] = tau_en_all
        s["en_rule_cov"], s["en_rule_err"] = selective(c, k, tau_en_all) if l != "en" else (None, None)
        cov, err, cov_e, err_e, ts = [], [], [], [], []
        oof_own, oof_en = np.full(n, np.nan), np.full(n, np.nan)
        for f in range(K):
            fit, test = folds != f, folds == f
            t_own = fit_threshold(c[fit], k[fit], a.alpha)
            t_en = fit_threshold(d["conf"][fit, en], d["correct"][fit, en], a.alpha)
            m = c[test] >= t_own; cov.append(m.sum()); err.append((1 - k[test][m]).sum())
            m = c[test] >= t_en; cov_e.append(m.sum()); err_e.append((1 - k[test][m]).sum())
            T_own = fit_temperature(d["probs"][fit, j], d["gold"][fit, j])
            T_en = fit_temperature(d["probs"][fit, en], d["gold"][fit, en])
            ts.append(T_own)
            oof_own[test] = apply_temperature(d["probs"][test, j], T_own)   # out-of-fold, pooled below
            oof_en[test] = apply_temperature(d["probs"][test, j], T_en)
        s["cf_own_n"] = int(sum(cov)); s["cf_en_n"] = int(sum(cov_e))
        s["cf_own_cov"] = sum(cov) / n; s["cf_own_err"] = sum(err) / sum(cov) if sum(cov) else float("nan")
        s["cf_en_cov"] = sum(cov_e) / n; s["cf_en_err"] = sum(err_e) / sum(cov_e) if sum(cov_e) else float("nan")
        s["ece_T_own"] = ece(oof_own, k); s["ece_T_en"] = ece(oof_en, k); s["T_own"] = float(np.mean(ts))
        s["gap_T_own"] = oof_own.mean() - k.mean(); s["gap_T_en"] = oof_en.mean() - k.mean()
        s["auroc_T_own"] = auroc(oof_own, k)
        # bootstrap intervals for the numbers the text leans on (resampling questions; the two
        # cross-fitting folds are resampled separately so no question sits in both)
        nb = min(a.boot, 1000)
        parts = [np.where(folds == f)[0] for f in range(K)]
        b_auc, b_en_err, b_en_cov, b_own_err, b_own_cov, b_both_err, b_flip_d, b_agree_auc = [], [], [], [], [], [], [], []
        ce, ke = d["conf"][:, en], d["correct"][:, en]
        for _ in range(nb):
            res = [rng.choice(q, len(q)) for q in parts]
            idx = np.concatenate(res)
            b_auc.append(auroc(c[idx], k[idx]))
            ne = no = we = wo = 0
            for f in range(K):
                test = res[f]; fit = np.concatenate([res[g] for g in range(K) if g != f])
                t_en = fit_threshold(ce[fit], ke[fit], a.alpha); t_own = fit_threshold(c[fit], k[fit], a.alpha)
                m = c[test] >= t_en; ne += m.sum(); we += (1 - k[test][m]).sum()
                m = c[test] >= t_own; no += m.sum(); wo += (1 - k[test][m]).sum()
            b_en_cov.append(ne / n); b_en_err.append(we / ne if ne else np.nan)
            b_own_cov.append(no / n); b_own_err.append(wo / no if no else np.nan)
            if "flip" in s:
                st = (d["rev_pred"][idx] == d["pred"][idx]).astype(float)
                b_flip_d.append((1 - st[:, j].mean()) - (1 - st[:, en].mean()))
                if l != "en":
                    ag = (d["pred"][idx, j] == d["pred"][idx, en]).astype(float)
                    bo = ag * st[:, j]
                    b_both_err.append(1 - k[idx][bo == 1].mean() if bo.any() else np.nan)
                    b_agree_auc.append(auroc(ag, k[idx]))
        s["auroc_ci"] = ci(np.array(b_auc)); s["cf_en_err_ci"] = ci(np.array(b_en_err)); s["cf_en_cov_ci"] = ci(np.array(b_en_cov))
        s["cf_own_err_ci"] = ci(np.array(b_own_err)); s["cf_own_cov_ci"] = ci(np.array(b_own_cov))
        if b_flip_d: s["flip_vs_en"] = s["flip"] - out_flip_en(d, en); s["flip_vs_en_ci"] = ci(np.array(b_flip_d))
        if b_both_err: s["both_err_ci"] = ci(np.array(b_both_err)); s["agree_auroc_ci"] = ci(np.array(b_agree_auc))
        out["langs"][l] = s
    return out


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


names = a.runs.split(",") if a.runs else list(dict.fromkeys(list(MODELS) + list(QUANT) + list(EXTRA)))
data = {n: load(n) for n in names}
data = {n: d for n, d in data.items() if d}
stats = {n: model_stats(d) for n, d in data.items()}
(ROOT / "results").mkdir(exist_ok=True); (ROOT / "report/fig").mkdir(parents=True, exist_ok=True)
json.dump(clean(stats), open(ROOT / "results/summary.json", "w"), indent=1)

pct = lambda x, d=0: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"
pp = lambda x: "–" if x is None or np.isnan(x) else (f"{100 * x:+.0f}" if abs(round(100 * x)) >= 1 else "0")
num = lambda x: "–" if x is None or np.isnan(x) else f"{x:.2f}"
disp = {**{k: f"Tiny Aya Global {v}" for k, v in QUANT.items()}, **MODELS, **EXTRA}
md = []

md.append("## Table 1. Accuracy, confidence and the gap between them\n")
md.append("| Model | Lang | Accuracy | Confidence | Gap (pp) [95% CI] | Gap minus English gap [95% CI] | ECE | AUROC | Answers at ≥90% | Wrong among those |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for n in [m for m in MODELS if m in stats] + [m for m in EXTRA if m in stats]:
    for l in LANGS:
        s = stats[n]["langs"][l]
        vs = "–" if l == "en" else f"{pp(s['gap_vs_en'])} [{pp(s['gap_vs_en_ci'][0])}, {pp(s['gap_vs_en_ci'][1])}]"
        md.append(f"| {disp[n]} (n={stats[n]['n']}) | {LNAME[l]} | {pct(s['acc'])} | {pct(s['conf'])} | {pp(s['gap'])} [{pp(s['gap_ci'][0])}, {pp(s['gap_ci'][1])}] | {vs} | {num(s['ece'])} | {num(s['auroc'])} | {pct(s['share_sure'])} | {pct(s['wrong_when_sure'])} (n={s['n_sure']}) |")

md.append(f"\n## Table 2. One confidence cut-off, five languages (target: at most {a.alpha:.0%} wrong among answered)\n")
md.append("| Model | Lang | English cut-off: answered | English cut-off: wrong | Own cut-off: answered | Own cut-off: wrong | ECE raw | ECE English temperature | ECE own temperature |")
md.append("|---|---|---|---|---|---|---|---|---|")
for n in [m for m in MODELS if m in stats]:
    for l in LANGS:
        s = stats[n]["langs"][l]
        md.append(f"| {disp[n]} | {LNAME[l]} | {pct(s['cf_en_cov'])} | {pct(s['cf_en_err'])} | {pct(s['cf_own_cov'])} | {pct(s['cf_own_err'])} | {num(s['ece'])} | {num(s['ece_T_en'])} | {num(s['ece_T_own'])} |")

md.append("\n## Table 3. Signals that need no labels in the target language\n")
md.append("| Model | Lang | Flips when options are reversed | Accuracy stable / flipped | Same answer as in English | Accuracy agree / disagree | Stable and agrees: answered | wrong | AUROC confidence | AUROC yes/no self-check | AUROC stated number | AUROC combined |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in [m for m in MODELS if m in stats]:
    for l in LANGS:
        s = stats[n]["langs"][l]; g = lambda k: s.get(k, float("nan"))
        md.append(f"| {disp[n]} | {LNAME[l]} | {pct(g('flip'))} | {pct(g('acc_stable'))} / {pct(g('acc_flipped'))} | {pct(g('agree_en'))} | {pct(g('acc_agree'))} / {pct(g('acc_disagree'))} | {pct(g('both_cov'))} | {pct(g('both_err'))} | {num(g('auroc'))} | {num(g('pyes_auroc'))} | {num(g('verbal_auroc'))} | {num(g('combo_auroc'))} |")

md.append("\n## Table 4. What the model says about itself\n")
md.append("| Model | Lang | Accuracy | Stated confidence (mean) | Most common stated number (share) | Says \"Yes, correct\" | Says Yes when wrong |")
md.append("|---|---|---|---|---|---|---|")
for n in [m for m in MODELS if m in stats]:
    for l in LANGS:
        s = stats[n]["langs"][l]; g = lambda k: s.get(k, float("nan"))
        mode = f"{s['verbal_mode']} ({pct(s['verbal_mode_share'])})" if "verbal_mode" in s else "–"
        md.append(f"| {disp[n]} | {LNAME[l]} | {pct(s['acc'])} | {pct(g('verbal_mean'))} | {mode} | {pct(g('pyes_says_yes'))} | {pct(g('pyes_yes_when_wrong'))} |")

qs = [q for q in QUANT if q in stats]
if qs:
    md.append("\n## Table 5. Tiny Aya Global at different precisions\n")
    md.append("| Precision | Lang | Accuracy | Confidence | Gap (pp) [95% CI] | ECE | AUROC | Flips when reversed |")
    md.append("|---|---|---|---|---|---|---|---|")
    for n in qs:
        for l in LANGS:
            s = stats[n]["langs"][l]
            md.append(f"| {QUANT[n]} (n={stats[n]['n']}) | {LNAME[l]} | {pct(s['acc'])} | {pct(s['conf'])} | {pp(s['gap'])} [{pp(s['gap_ci'][0])}, {pp(s['gap_ci'][1])}] | {num(s['ece'])} | {num(s['auroc'])} | {pct(s.get('flip', float('nan')))} |")
open(ROOT / "results/tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "savefig.dpi": 300, "savefig.bbox": "tight"})
main = [m for m in MODELS if m in stats]


def fig_gap():
    """Fig 1: dumbbell per model. Accuracy vs mean confidence, per language."""
    if not main: return
    fig, axes = plt.subplots(1, len(main), figsize=(1.9 * len(main) + 0.9, 2.3), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], main):
        for y, l in enumerate(LANGS[::-1]):
            s = stats[n]["langs"][l]
            ax.plot([s["acc"], s["conf"]], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            ax.plot(s["acc_ci"], [y, y], color=INK, lw=0.8, zorder=2)
            ax.scatter([s["conf"]], [y], s=46, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1)
            ax.scatter([s["acc"]], [y], s=20, color=INK, zorder=4, edgecolor=SURF, linewidth=0.6)   # smaller, on top: both show when they coincide
            ax.text(1.03, y, pp(s["gap"]).replace("-", "\u2212"), va="center", ha="left", fontsize=7.5, color=MUTED)
        ax.axvline(0.25, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(0.2, 1.0); ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["25", "50", "75", "100"])
        ax.set_ylim(-0.6, len(LANGS) - 0.4); ax.set_yticks(range(len(LANGS))); ax.set_yticklabels([LNAME[l] for l in LANGS[::-1]])
        ax.tick_params(length=0); ax.set_title(MODELS[n], loc="left"); ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0][0].scatter([], [], s=20, color=INK, label="accuracy (%)"); axes[0][0].scatter([], [], s=46, color=ORANGE, label="mean confidence (%)")
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.09), handletextpad=0.3, columnspacing=1.5)
    fig.text(0.995, 0.93, "gap (pp)", ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(ROOT / "report/fig/fig1_gap.png"); plt.close(fig)


def fig_reliability():
    """Fig 2: reliability curves, English vs the four African languages pooled, per model."""
    if not main: return
    fig, axes = plt.subplots(1, len(main), figsize=(1.9 * len(main) + 0.6, 2.2), sharey=True, squeeze=False)
    edges = np.array([0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.97, 1.0001])
    for ax, n in zip(axes[0], main):
        d = data[n]
        ax.plot([0.25, 1], [0.25, 1], color=GRID, lw=1, zorder=0)
        for cols, color, lab in (([0], BLUE, "English"), ([1, 2, 3, 4], ORANGE, "Swahili, Hausa, Igbo, Yoruba")):
            c = d["conf"][:, cols].ravel(); k = d["correct"][:, cols].ravel()
            xs, ys, ss = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (c >= lo) & (c < hi)
                if m.sum() >= 10:
                    xs.append(c[m].mean()); ys.append(k[m].mean()); ss.append(m.mean())
            ax.plot(xs, ys, color=color, lw=1.6, marker="o", ms=4, mec=SURF, mew=0.8, label=lab)
        ax.set_xlim(0.22, 1.02); ax.set_ylim(0.0, 1.02); ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["25", "50", "75", "100"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.set_title(MODELS[n], loc="left"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.set_xlabel("confidence in its answer (%)")
    axes[0][0].set_ylabel("answers that are correct (%)")
    h, lb = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.24))
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(ROOT / "report/fig/fig2_reliability.png"); plt.close(fig)


def fig_threshold():
    """Fig 2: answering only above a confidence cut-off. Top: wrong among answered. Bottom: share answered."""
    if not main: return
    fig, axes = plt.subplots(2, len(main), figsize=(1.9 * len(main) + 0.7, 2.6), sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.16, "wspace": 0.1})
    x = np.arange(len(LANGS)); w = 0.36
    for col, n in enumerate(main):
        top, bot = axes[0][col], axes[1][col]
        for dx, color, err, cov, cnt, cik, lab in ((-w / 2 - 0.01, ORANGE, "cf_en_err", "cf_en_cov", "cf_en_n", "cf_en_err_ci", "cut-off tuned on English"),
                                                   (w / 2 + 0.01, BLUE, "cf_own_err", "cf_own_cov", "cf_own_n", "cf_own_err_ci", "cut-off tuned on the language itself")):
            for xi, l in zip(x, LANGS):
                s = stats[n]["langs"][l]
                bot.bar(xi + dx, s[cov], w, color=color)
                if s[cnt] >= 10 and not np.isnan(s[err]):       # too few answers to estimate an error rate otherwise
                    top.bar(xi + dx, s[err], w, color=color, label=lab if (xi == 0 and col == 0) else None)
                    lo, hi = s[cik]
                    if not (np.isnan(lo) or np.isnan(hi)):
                        top.plot([xi + dx, xi + dx], [lo, min(hi, 0.84)], color=INK, lw=0.7, zorder=4)
                else:
                    top.text(xi + dx, 0.012, "–", ha="center", va="bottom", fontsize=7, color=MUTED)
        top.axhline(a.alpha, color=INK, lw=0.8, ls=(0, (3, 2)), label=f"target: {a.alpha:.0%} wrong" if col == 0 else None)
        top.set_ylim(0, 0.85); top.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); bot.set_ylim(0, 1.0); bot.set_yticks([0, 0.5, 1.0])
        for ax in (top, bot):
            ax.set_xticks(x); ax.tick_params(length=0); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        top.set_xticklabels([]); bot.set_xticklabels(LANGS); top.set_title(MODELS[n], loc="left")
    axes[0][0].set_ylabel("wrong among\nanswered"); axes[1][0].set_ylabel("questions\nanswered")
    h, lb = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(ROOT / "report/fig/fig3_threshold.png"); plt.close(fig)


def fig_quant():
    """Fig 4: Tiny Aya accuracy and confidence by precision, per language (small multiples)."""
    q = [n for n in QUANT if n in stats]
    if len(q) < 2: return
    fig, axes = plt.subplots(1, len(LANGS), figsize=(7.6, 1.5), sharey=True)
    for ax, l in zip(axes, LANGS):
        x = np.arange(len(q))
        ax.plot(x, [stats[n]["langs"][l]["acc"] for n in q], color=INK, lw=1.6, marker="o", ms=4, mec=SURF, mew=0.8, label="accuracy")
        ax.plot(x, [stats[n]["langs"][l]["conf"] for n in q], color=ORANGE, lw=1.6, marker="o", ms=4, mec=SURF, mew=0.8, label="mean confidence")
        ax.set_xticks(x); ax.set_xticklabels([QUANT[n].replace("-bit", "") for n in q]); ax.set_title(LNAME[l], loc="left")
        ax.set_ylim(0.2, 1.0); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["25", "50", "75", "100"])
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=0)
    axes[0].set_ylabel("%"); axes[2].set_xlabel("bits per weight")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.36))
    fig.savefig(ROOT / "report/fig/fig4_quant.png"); plt.close(fig)


for f in (fig_gap, fig_reliability, fig_threshold, fig_quant):
    try:
        f()
    except Exception as e:
        print("figure failed:", f.__name__, repr(e))
print("\nfigures -> report/fig/")
