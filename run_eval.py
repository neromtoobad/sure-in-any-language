"""Run one GGUF model over the parallel Belebele items and record how sure it is.

usage: python run_eval.py <gguf> <name> [--n 300] [--passes orig,ptrue,verbal,rev] [--langs en,sw,yo,ha,ig]

For every (item, language) it records:
  orig    next-token probabilities over the option letters A-D (the model's internal confidence)
  ptrue   probability it says "Yes" when asked whether its own answer is correct
  verbal  the number the model says when asked how likely its own answer is to be correct
  rev     like orig, with the options listed in reverse order (does the answer survive a reshuffle?)

Instructions are always in English; only the passage, question and options change language.
Decoding is greedy. Writes runs/<name>/answers.jsonl (resumable) and runs/<name>/meta.json.

Speed: all passes for one item run back to back on the same server slot with the prompt cache
on, so each step only costs the tokens that differ from the step before it.
"""
import argparse, json, math, os, queue, re, signal, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
SERVER = os.environ.get("LLAMA_SERVER", str(ROOT.parent / "lexsi/bin/llama-b12403/llama-server"))
LETTERS = "ABCD"

ap = argparse.ArgumentParser()
ap.add_argument("gguf"); ap.add_argument("name")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--passes", default="orig,ptrue,verbal,rev")
ap.add_argument("--langs", default="en,sw,yo,ha,ig")
ap.add_argument("--parallel", type=int, default=4)
ap.add_argument("--port", type=int, default=8731)
ap.add_argument("--nice", type=int, default=0)
a = ap.parse_args()
passes = a.passes.split(","); langs = a.langs.split(",")

out_dir = ROOT / "runs" / a.name; out_dir.mkdir(parents=True, exist_ok=True)
out_f = out_dir / "answers.jsonl"
done = {}
if out_f.exists():
    for l in out_f.open(encoding="utf-8"):
        r = json.loads(l); done[(r["uid"], r["lang"], r["pass"])] = r

items = [json.loads(l) for l in open(ROOT / "data/items.jsonl", encoding="utf-8")]
items = [r for r in items if r["rank"] < a.n and r["lang"] in langs]

ASK = ("Read the passage and answer the question. "
       "Reply with only the letter (A, B, C or D) of the correct answer.\n\n"
       "Passage: {passage}\n\nQuestion: {question}\n\n{opts}\n\nAnswer:")
ASK_CONF = ("What is the probability, from 0 to 100, that the answer you just gave is correct? "
            "Reply with only the number.")
ASK_SURE = "Is the answer you just gave correct? Reply with only Yes or No."


def question_text(it, order):
    """order: list of original option indices (0-3) in display order."""
    opts = "\n".join(f"{LETTERS[i]}. {it['options'][o]}" for i, o in enumerate(order))
    return ASK.format(passage=it["passage"], question=it["question"], opts=opts)


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def templated(messages):
    return post("/apply-template", {"messages": messages})["prompt"]


def complete(prompt, slot, n_predict, n_probs=0):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": True, "id_slot": slot}
    if n_probs:
        body["n_probs"] = n_probs
    return post("/completion", body)


def first_token_probs(r):
    top = r["completion_probabilities"][0]
    out = []
    for c in top.get("top_logprobs") or top.get("top_probs") or top.get("probs"):
        p = math.exp(c["logprob"]) if "logprob" in c else c["prob"]
        out.append(((c.get("token") or "").strip(), p))
    return out


def answer_pass(it, name, slot):
    order = [0, 1, 2, 3] if name == "orig" else [3, 2, 1, 0]
    r = complete(templated([{"role": "user", "content": question_text(it, order)}]), slot, 1, 40)
    mass = [0.0] * 4
    for tok, p in first_token_probs(r):
        if len(tok) == 1 and tok in LETTERS:
            mass[LETTERS.index(tok)] += p
    total = sum(mass)
    probs = [m / total for m in mass] if total > 0 else [0.25] * 4
    pick = max(range(4), key=lambda i: probs[i])
    by_option = [0.0] * 4          # probabilities re-indexed by ORIGINAL option (0-3)
    for i, o in enumerate(order):
        by_option[o] = probs[i]
    return {"probs": [round(p, 5) for p in by_option], "letter_mass": round(total, 5),
            "pred": order[pick] + 1, "conf": round(probs[pick], 5), "ntok": r.get("tokens_evaluated")}


def followup(it, orig, ask):
    q = question_text(it, [0, 1, 2, 3])
    return templated([{"role": "user", "content": q}, {"role": "assistant", "content": LETTERS[orig["pred"] - 1]},
                      {"role": "user", "content": ask}])


def ptrue_pass(it, orig, slot):
    """Self-check: probability the model says Yes when asked if its own answer is correct."""
    r = complete(followup(it, orig, ASK_SURE), slot, 1, 40)
    yes = no = 0.0
    for tok, p in first_token_probs(r):
        if tok.lower() == "yes": yes += p
        elif tok.lower() == "no": no += p
    return {"p_yes": round(yes / (yes + no), 5) if yes + no > 0 else None, "yn_mass": round(yes + no, 5),
            "pred": orig["pred"]}


def verbal_pass(it, orig, slot):
    """Stated confidence about the answer the model actually gave in the orig pass."""
    raw = complete(followup(it, orig, ASK_CONF), slot, 8)["content"].strip()
    m = re.search(r"\d+(?:\.\d+)?", raw)
    val = float(m.group()) if m else None
    if val is not None and (val < 0 or val > 100):
        val = None
    return {"verbal": val, "verbal_raw": raw[:40], "pred": orig["pred"]}


lock = threading.Lock()
out = out_f.open("a", encoding="utf-8")
slots = queue.Queue()
for s in range(a.parallel):
    slots.put(s)


def save(it, name, rec):
    rec.update(uid=it["uid"], lang=it["lang"], gold=it["gold"]); rec["pass"] = name
    with lock:
        done[(it["uid"], it["lang"], name)] = rec
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()


def work(job):
    it, names = job
    slot = slots.get()
    try:
        for name in names:   # same slot, in order, so follow-ups reuse the cached question
            if (it["uid"], it["lang"], name) in done:
                continue
            if name in ("orig", "rev"):
                rec = answer_pass(it, name, slot)
            else:
                orig = done.get((it["uid"], it["lang"], "orig"))
                if orig is None:
                    continue
                rec = ptrue_pass(it, orig, slot) if name == "ptrue" else verbal_pass(it, orig, slot)
            save(it, name, rec)
    except Exception as e:  # keep going; the row is retried on the next run
        return f"{it['uid']} {it['lang']}: {e}"
    finally:
        slots.put(slot)
    return None


# One chain per item: every step shares its prompt prefix with the step before it (rev shares the
# passage and question), so the server only evaluates the new tokens.
chain = [n for n in ("orig", "ptrue", "verbal", "rev") if n in passes]
stages = [("chain", chain)]
if not any((it["uid"], it["lang"], n) not in done for it in items for n in passes):
    print(f"[{a.name}] nothing to do"); sys.exit(0)

pre = ["nice", "-n", str(a.nice)] if a.nice else []
srv = subprocess.Popen(pre + [SERVER, "-m", a.gguf, "--port", str(a.port), "-ngl", "99",
                              "-c", str(4096 * a.parallel), "--parallel", str(a.parallel), "--no-webui"],
                       stdout=open(out_dir / "server.log", "w"), stderr=subprocess.STDOUT)


def stop(*_):
    srv.terminate()
    os._exit(1)   # hard exit: worker threads would otherwise keep the process alive


signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
try:
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2); break
        except Exception:
            if srv.poll() is not None or time.time() - t0 > 600:
                sys.exit(f"[{a.name}] server failed to start, see {out_dir}/server.log")
            time.sleep(1)
    props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/props", timeout=10).read())
    json.dump({"name": a.name, "gguf": os.path.basename(a.gguf), "gguf_bytes": os.path.getsize(a.gguf),
               "n_items": a.n, "langs": langs, "passes": passes, "ask": ASK, "ask_conf": ASK_CONF,
               "ask_sure": ASK_SURE, "chat_template": props.get("chat_template", "")[:4000],
               "server": "llama.cpp b12403"}, open(out_dir / "meta.json", "w"), indent=1)
    for label, names in stages:
        jobs = [(it, names) for it in items if any((it["uid"], it["lang"], n) not in done for n in names)]
        if not jobs:
            continue
        t0 = time.time(); errs = 0
        with ThreadPoolExecutor(a.parallel) as ex:
            for i, e in enumerate(ex.map(work, jobs), 1):
                if e:
                    errs += 1
                    if errs <= 3: print(f"[{a.name}] error {e}", flush=True)
                if i % 100 == 0 or i == len(jobs):
                    print(f"[{a.name}] {label} {i}/{len(jobs)}  {i / (time.time() - t0):.2f} items/s  errors {errs}", flush=True)
finally:
    srv.terminate()
    try: srv.wait(20)
    except Exception: srv.kill()
