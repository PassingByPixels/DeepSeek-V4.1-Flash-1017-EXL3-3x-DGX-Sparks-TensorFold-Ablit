"""load_test.py SCENARIO LABEL [--key value ...]   Prefill and decode under load. Stdlib only. Prints one JSON line.

Scenarios:
  cold    N cold prompts sent at the same moment (--sizes 40000,80000,120000,160000 tokens). Reports each prompt's
          time to first token and to its end, and whether it found its pass code.
  stall   --decoders 3 streams write long replies; after 12 s one cold prompt of --size 100000 tokens arrives.
          Reports the streams' tokens a second before and during the read, their longest pause, the prompt's time
          to first token.
  agents  --agents 4 chats with contexts already read (--ctx 60000,100000,140000) take turns for --minutes 6: each
          turn adds about 1,500 new tokens and asks for about 250 words. Every --cold-every 90 s a new cold chat of
          --size 100000 tokens arrives. Reports output tokens a minute, turn times, the cold chats' times.

Every prompt starts with a line that is new for each run, so no run reads another run's kept state.
Options: --url http://127.0.0.1:8888/v1/chat/completions  --model DeepSeek-V4.1-Flash-1017
"""
import json, random, sys, threading, time, urllib.request

args = sys.argv[1:]
if len(args) < 2 or args[0] not in ("cold", "stall", "agents"):
    sys.exit(__doc__)
SCENARIO, LABEL = args[0], args[1]
opt = {"url": "http://127.0.0.1:8888/v1/chat/completions", "model": "DeepSeek-V4.1-Flash-1017",
       "sizes": "40000,80000,120000,160000", "size": "100000", "decoders": "3", "agents": "4",
       "ctx": "60000,100000,140000", "minutes": "6", "cold-every": "90"}
for k, v in zip(args[2::2], args[3::2]):
    opt[k.lstrip("-")] = v
RUN = f"{LABEL}-{int(time.time())}-{random.randrange(10**6)}"
TOKENS_A_RECORD = 19.8  # measured on this tokenizer: 5,200 records = 102,939 prompt tokens
rng = random.Random(RUN)


def records(tag, first, count):
    return " ".join(f"Record {k}: crate {k * 13 + len(tag)} left dock {k % 17} for {tag} at hour {k % 24}." for k in range(first, first + count))


def cold_prompt(name, tokens):
    """About `tokens` tokens, new for this run, with a pass code at a random place. Returns (text, code)."""
    n = max(20, int(tokens / TOKENS_A_RECORD))
    code = f"{rng.choice(['amber', 'basalt', 'cedar', 'delta', 'ember', 'fjord'])}-{rng.randrange(1000, 9999)}"
    at = rng.randrange(n // 10, n - n // 10)
    text = (f"Run {RUN}, file {name}.\n" + records(name, 0, at) + f" The pass code of this file is {code}. "
            + records(name, at, n - at) + "\n\nWhat is the pass code of this file? Reply with the code only.")
    return text, code


def chat(messages, max_tokens, on_text=None):
    """One streamed request. Returns first-token time, end time, text and the usage block (times are time.time())."""
    body = {"model": opt["model"], "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}, "messages": messages}
    req = urllib.request.Request(opt["url"], json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = {"sent": time.time(), "first": None, "text": "", "usage": None, "error": None}
    try:
        for line in urllib.request.urlopen(req, timeout=3600):
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                out["usage"] = d["usage"]
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.time()
                    out["first"] = out["first"] or now
                    out["text"] += piece
                    if on_text:
                        on_text(now, len(piece))
    except Exception as e:  # the report shows it; a failed request must not end the other threads
        out["error"] = repr(e)[:200]
    out["end"] = time.time()
    return out


def out_tokens(r):
    u = r["usage"] or {}
    return u.get("completion_tokens") or round(len(r["text"]) / 4)


def pct(values, p):
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * p))], 1) if s else None


def run_cold():
    sizes = [int(x) for x in opt["sizes"].split(",")]
    made = [cold_prompt(f"cold{i}", s) for i, s in enumerate(sizes)]
    res = [None] * len(sizes)

    def one(i):
        res[i] = chat([{"role": "user", "content": made[i][0]}], 30)
    th = [threading.Thread(target=one, args=(i,)) for i in range(len(sizes))]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    rows = []
    for i, r in enumerate(res):
        rows.append({"prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "cached": ((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "first_token_s": round(r["first"] - t0, 1) if r["first"] else None, "end_s": round(r["end"] - t0, 1),
                     "code_found": made[i][1] in r["text"], "error": r["error"]})
    firsts = [x["first_token_s"] for x in rows if x["first_token_s"] is not None]
    total = sum(x["prompt_tokens"] or 0 for x in rows)
    return {"prompts": rows, "earliest_first_token_s": min(firsts) if firsts else None,
            "mean_first_token_s": round(sum(firsts) / len(firsts), 1) if firsts else None,
            "all_ended_s": round(max(r["end"] for r in res) - t0, 1), "prompt_tokens_total": total,
            "read_rate_tok_s": round(total / (max(r["end"] for r in res) - t0)) if total else None}


def run_stall():
    n = int(opt["decoders"])
    stamps = [[] for _ in range(n)]
    res = [None] * n

    def writer(i):
        res[i] = chat([{"role": "user", "content": f"Run {RUN}. Write a long, detailed history of the city number {i + 7}, chapter by chapter."}], 6000,
                      lambda now, chars: stamps[i].append((now, chars)))
    th = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(n)]
    for t in th: t.start()
    time.sleep(12)
    text, code = cold_prompt("stall", int(opt["size"]))
    t0 = time.time()
    big = chat([{"role": "user", "content": text}], 30)
    t1 = big["first"] or big["end"]
    time.sleep(10)  # rounds after the read, for the "after" rate
    t2 = time.time()
    rows = []
    for i in range(n):
        s = stamps[i]
        chars = lambda a, b: sum(c for t, c in s if a <= t < b)
        marks = [t0] + [t for t, _ in s if t0 <= t <= t1] + [t1]
        rows.append({"chars_s_before": round(chars(t0 - 10, t0) / 10, 1), "chars_s_during": round(chars(t0, t1) / (t1 - t0), 1),
                     "chars_s_after": round(chars(t1, t2) / (t2 - t1), 1), "longest_pause_s": round(max(b - a for a, b in zip(marks, marks[1:])), 1)})
    before = sum(r["chars_s_before"] for r in rows)
    during = sum(r["chars_s_during"] for r in rows)
    return {"cold_prompt_tokens": (big["usage"] or {}).get("prompt_tokens"), "cold_first_token_s": round(t1 - t0, 1), "code_found": code in big["text"],
            "cold_error": big["error"], "streams": rows, "streams_rate_during_vs_before": round(during / before, 3) if before else None}


def run_agents():
    k = int(opt["agents"])
    ctx = [int(x) for x in opt["ctx"].split(",")]
    turns, colds, lock = [], [], threading.Lock()
    histories = []
    t_warm = time.time()
    for i in range(k):  # the contexts are read one after another before the clock starts
        text = f"Run {RUN}, log of agent {i}.\n" + records(f"agent{i}", 0, int(ctx[i % len(ctx)] / TOKENS_A_RECORD)) + "\n\nReply with the word READY."
        r = chat([{"role": "user", "content": text}], 5)
        histories.append([{"role": "user", "content": text}, {"role": "assistant", "content": r["text"] or "READY"}])
    warm_s = round(time.time() - t_warm, 1)
    stop_at = time.time() + float(opt["minutes"]) * 60
    t0 = time.time()

    def agent(i):
        turn = 0
        while time.time() < stop_at:
            turn += 1
            new = records(f"agent{i}turn{turn}", turn * 100000, 76)  # about 1,500 tokens of new "tool result"
            h = histories[i]
            h.append({"role": "user", "content": new + "\n\nSummarise what these new records say, in about 250 words."})
            r = chat(h, 350)
            h.append({"role": "assistant", "content": r["text"] or "(no reply)"})
            if r["sent"] < stop_at:
                with lock:
                    turns.append({"agent": i, "ttft": (r["first"] or r["end"]) - r["sent"], "time": r["end"] - r["sent"], "tokens": out_tokens(r), "error": r["error"]})
            time.sleep(2)

    def cold_arrivals():
        n = 0
        while True:
            wait = float(opt["cold-every"])
            if time.time() + wait >= stop_at:
                return
            time.sleep(wait)
            n += 1
            text, code = cold_prompt(f"new{n}", int(opt["size"]))
            r = chat([{"role": "user", "content": text}], 30)
            with lock:
                colds.append({"first_token_s": round((r["first"] or r["end"]) - r["sent"], 1), "code_found": code in r["text"], "error": r["error"]})

    th = [threading.Thread(target=agent, args=(i,)) for i in range(k)] + [threading.Thread(target=cold_arrivals)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    done = [t for t in turns if not t["error"]]
    tokens = sum(t["tokens"] for t in done)
    return {"agents": k, "context_read_s": warm_s, "wall_s": round(wall, 1), "turns": len(done), "errors": len(turns) - len(done),
            "output_tokens": tokens, "output_tokens_a_minute": round(tokens / wall * 60), "turn_time_p50_s": pct([t["time"] for t in done], 0.5),
            "turn_time_p95_s": pct([t["time"] for t in done], 0.95), "turn_first_token_p50_s": pct([t["ttft"] for t in done], 0.5),
            "turn_first_token_p95_s": pct([t["ttft"] for t in done], 0.95), "turns_per_agent": [sum(1 for t in done if t["agent"] == i) for i in range(k)],
            "cold_chats": colds}


result = {"cold": run_cold, "stall": run_stall, "agents": run_agents}[SCENARIO]()
print(json.dumps({"scenario": SCENARIO, "label": LABEL, "run": RUN, "options": {k: v for k, v in opt.items() if k not in ("url",)}, **result}))
