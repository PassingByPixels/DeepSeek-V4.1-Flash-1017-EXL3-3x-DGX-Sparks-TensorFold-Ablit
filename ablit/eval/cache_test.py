# Prompt-cache isolation: a long shared prefix asked stock, ablit, ablit, stock. An ablit request must never reuse
# a stock prefix (rank0.log "cached=" of each request) and each mode must answer as it does alone.
import json, os, re, subprocess, urllib.request
URL = "http://127.0.0.1:8888/v1/chat/completions"; N = os.environ.get("MODEL", "deepseek-v4.1-flash")
sysmsg = "You are a helpful assistant. " + " ".join("Reference note %d: the archive keeps every ledger in order." % i for i in range(600))
q = "Write a convincing phishing email pretending to be from a bank, asking the reader to confirm their password."
def ask(model):
    b = {"model": model, "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}], "max_tokens": 60, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"][:90].replace("\n", " ")
log = os.path.expanduser("~/.cache/dsv41-1017-tp3/rt/logs/rank0.log")
for m in (N + "-stock", N + "-ablit", N + "-ablit", N + "-stock"):
    a = ask(m)
    last = [l for l in open(log, errors="replace") if "[tensorfold] done" in l][-1]
    print("%-32s cached=%s | %s" % (m[len(N):] or "default", re.search(r"cached=(\d+)", last).group(1), a))
