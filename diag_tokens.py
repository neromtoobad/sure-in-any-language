"""Diagnostic: which first tokens does a model put probability on, besides the letters A-D?

usage: python diag_tokens.py <gguf> [--n 20] [--langs en,yo] [--port 8732]
Prints, per language, the average first-token probability of each token over the first n questions.
"""
import argparse, json, math, os, subprocess, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = os.environ.get("LLAMA_SERVER", str(ROOT.parent / "lexsi/bin/llama-b12403/llama-server"))
ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--langs", default="en,yo"); ap.add_argument("--port", type=int, default=8732)
a = ap.parse_args()
ASK = ("Read the passage and answer the question. "
       "Reply with only the letter (A, B, C or D) of the correct answer.\n\n"
       "Passage: {passage}\n\nQuestion: {question}\n\n{opts}\n\nAnswer:")
items = [json.loads(l) for l in open(ROOT / "data/items.jsonl", encoding="utf-8")]


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


srv = subprocess.Popen([SERVER, "-m", a.gguf, "--port", str(a.port), "-ngl", "99", "-c", "4096", "--no-webui"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    t0 = time.time()
    while True:
        try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2); break
        except Exception:
            if srv.poll() is not None or time.time() - t0 > 300: sys.exit("server failed")
            time.sleep(1)
    for lang in a.langs.split(","):
        tot = defaultdict(float); first = defaultdict(int); k = 0; gen = []
        for it in [r for r in items if r["lang"] == lang and r["rank"] < a.n]:
            opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(it["options"]))
            prompt = post("/apply-template", {"messages": [{"role": "user", "content": ASK.format(passage=it["passage"], question=it["question"], opts=opts)}]})["prompt"]
            r = post("/completion", {"prompt": prompt, "n_predict": 6, "temperature": 0, "n_probs": 40})
            top = r["completion_probabilities"][0]
            cands = top.get("top_logprobs") or top.get("top_probs") or top.get("probs")
            for c in cands:
                p = math.exp(c["logprob"]) if "logprob" in c else c["prob"]
                tot[repr(c.get("token"))] += p
            first[repr(cands[0].get("token"))] += 1; k += 1; gen.append(r["content"])
        print(f"\n[{lang}] {k} questions. mean first-token probability by token:")
        for tok, p in sorted(tot.items(), key=lambda x: -x[1])[:14]:
            print(f"   {tok:>12}  {p / k:.3f}")
        print("   most likely first token:", dict(sorted(first.items(), key=lambda x: -x[1])))
        print("   generated text (first 6 tokens), examples:", [g for g in gen[:8]])
finally:
    srv.terminate()
