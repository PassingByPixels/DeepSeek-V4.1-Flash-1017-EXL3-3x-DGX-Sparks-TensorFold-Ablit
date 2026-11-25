"""tool_probe.py flat|nested|ops [URL] [MODEL] : how do a streamed tool call's ARGUMENTS arrive? Stdlib only.

One streamed chat completion with one tool. Prints the count of argument pieces, the largest piece, the longest
gap between pieces, and whether the joined arguments are JSON with the expected keys. No key, no user data.
  flat    write_note {title, text}                         string values at the top level
  nested  folio {tool, args: {title, text}}                the text sits inside an object value
  ops     write_blocks {ops: [{op, text}, ...]}            the text sits inside a list of objects
"""
import json, sys, time, urllib.request

mode = sys.argv[1] if len(sys.argv) > 1 else "flat"
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8888/v1/chat/completions"
model = sys.argv[3] if len(sys.argv) > 3 else "DeepSeek-V4.1-Flash-1017"
TOOLS = {
    "flat": ("write_note", {"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}}, "required": ["title", "text"]},
             "Call write_note once. Its text must be a 150 word paragraph about lighthouses."),
    "nested": ("folio", {"type": "object", "properties": {"tool": {"type": "string"}, "args": {"type": "object"}}, "required": ["tool", "args"]},
               "Call folio once with tool \"write_chunk\" and args an object with a title and a text. The text must be a 150 word paragraph about lighthouses."),
    "ops": ("write_blocks", {"type": "object", "properties": {"ops": {"type": "array", "items": {"type": "object"}}}, "required": ["ops"]},
            "Call write_blocks once. ops must be a list of three objects, each with op \"set\" and a text of about 50 words about lighthouses."),
}
name, params, ask = TOOLS[mode]
body = {"model": model, "stream": True, "max_tokens": 900, "temperature": 0, "chat_template_kwargs": {"thinking": False},
        "messages": [{"role": "user", "content": ask}],
        "tools": [{"type": "function", "function": {"name": name, "description": "Save it.", "parameters": params}}], "tool_choice": "auto"}
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
pieces, finish = [], None
with urllib.request.urlopen(req, timeout=300) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        d = json.loads(line[5:])
        for c in d.get("choices") or []:
            finish = c.get("finish_reason") or finish
            for call in (c.get("delta") or {}).get("tool_calls") or []:
                a = (call.get("function") or {}).get("arguments")
                if a:
                    pieces.append((time.time() - t0, a))
joined = "".join(a for _, a in pieces)
try:
    parsed = json.loads(joined)
    ok = isinstance(parsed, dict) and all(k in parsed for k in params["required"])
except ValueError:
    ok = False
times = [t for t, _ in pieces]
gaps = [b - a for a, b in zip(times, times[1:])]
print(json.dumps({"mode": mode, "pieces": len(pieces), "characters": len(joined), "largest_piece": max((len(a) for _, a in pieces), default=0),
                  "first_piece_s": round(times[0], 2) if times else None, "last_piece_s": round(times[-1], 2) if times else None,
                  "longest_gap_s": round(max(gaps), 2) if gaps else None, "arguments_are_json_with_the_keys": ok, "finish": finish}))
