"""
ablit_fetch.py REPO REV OUT LAYER...: download only `layers.N.attn.wo_b.weight` and `.scale` of the given layers from
a Hugging Face safetensors checkpoint at a pinned revision, by HTTP range requests (about 42 MB a layer instead of
whole shards). Standard library only. HF_TOKEN (or ~/.cache/huggingface/token) is sent when present.

Writes OUT/L<N>.weight.bin, OUT/L<N>.scale.bin and OUT/L<N>.json (dtype and shape of both). Files already present
with the right size are skipped. Every range is checked against the byte length the shard's header gives.
"""
import json, os, struct, sys, time, urllib.request

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


def get(url, a = None, b = None, tries = 6):
    h = {"Authorization": f"Bearer {token()}"} if token() else {}
    if a is not None:
        h["Range"] = f"bytes={a}-{b}"
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers = h), timeout = 600).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    repo, rev, out = sys.argv[1:4]
    layers = [int(x) for x in sys.argv[4:]]
    os.makedirs(out, exist_ok = True)
    index = json.loads(get(f"{HUB}/{repo}/resolve/{rev}/model.safetensors.index.json"))["weight_map"]
    headers = {}
    for n in layers:
        shard = index[f"layers.{n}.attn.wo_b.weight"]
        url = f"{HUB}/{repo}/resolve/{rev}/{shard}"
        if shard not in headers:
            size = struct.unpack("<Q", get(url, 0, 7))[0]
            headers[shard] = (8 + size, json.loads(get(url, 8, 8 + size - 1)))
        base, h = headers[shard]
        meta = {}
        for part in ("weight", "scale"):
            m = h[f"layers.{n}.attn.wo_b.{part}"]
            a, b = m["data_offsets"]
            path = f"{out}/L{n}.{part}.bin"
            meta[part] = {"dtype": m["dtype"], "shape": m["shape"]}
            if os.path.exists(path) and os.path.getsize(path) == b - a:
                continue
            data = get(url, base + a, base + b - 1)
            if len(data) != b - a:
                sys.exit(f"layer {n} {part}: got {len(data)} bytes, want {b - a}")
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        json.dump(meta, open(f"{out}/L{n}.json", "w"))
        print(f"   layer {n}: wo_b from {shard}", flush = True)


if __name__ == "__main__":
    main()
