#!/usr/bin/env python3
"""mmlu.py LABEL: 240 fixed MMLU-Pro questions (seed 7), thinking off, 8 concurrent, letter answer. this folder: mmlu-LABEL.json"""
import json, os, re, sys, threading, urllib.request
H = os.path.dirname(os.path.abspath(__file__)); URL = 'http://127.0.0.1:8888/v1/chat/completions'
Q = json.load(open(H + '/mmlupro240.json')); L = 'ABCDEFGHIJ'
def ask(text):
    body = {"model": os.environ.get("MODEL", "deepseek-v4.1-flash"), "messages": [{"role": "user", "content": text}], "max_tokens": 900, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"].get("content") or ""
res = [None] * len(Q); lock = threading.Lock(); it = iter(range(len(Q)))
def w():
    while True:
        with lock: i = next(it, None)
        if i is None: return
        q = Q[i]; opts = "\n".join(f"{L[k]}. {o}" for k, o in enumerate(q["opts"]))
        c = ask(f"{q['q']}\n\n{opts}\n\nThink briefly, then end with a line 'Answer: X' where X is the letter.")
        m = re.findall(r"Answer:\s*\(?([A-J])\)?", c); got = m[-1] if m else ""
        res[i] = (got == q["a"], q["cat"])
ts = [threading.Thread(target=w) for _ in range(8)]; [t.start() for t in ts]; [t.join() for t in ts]
ok = sum(r[0] for r in res); json.dump({"label": sys.argv[1], "ok": ok, "n": len(Q), "res": res}, open(f"{H}/mmlu-{sys.argv[1]}.json", "w"))
print(f"{sys.argv[1]}: MMLU-Pro {ok}/{len(Q)} = {ok/len(Q):.3f}")
