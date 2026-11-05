"""
ablit_direction.py STOCK_DIR SIDECAR OUT LAMBDA LAYER...: the refusal direction of drowzeys' Keys overlay, for the
run-time edit (ablit/ablit-runtime.patch). Needs numpy (run in the recipe's PyTorch image).

The overlay's FP8 sidecar holds `attn.wo_b` of layers 10-35 after the edit W' = W - lam r r^T W, with one direction r
(5120-d, the hidden size) for all layers. For each layer this dequantises the stock and the edited FP8 weights
(e4m3 values, UE8M0 32x32 block scales), takes the top output-side singular vector of the difference, aligns the
signs, and averages them into one unit direction. It writes OUT in the runtime's format ("ABL1", u32 hidden,
u32 count, then per layer u32 layer, f32 lambda, f32[hidden] r) and OUT.json with the checks:

  - sigma1 / sigma2 of each difference (the edit is rank 1 when this is large; ~15 on the 1017 overlay),
  - each layer's direction against the mean (cosine; ~0.9999),
  - the edit strength fitted on the FP8 file (~3.3-3.4; the overlay's stated lambda is 3.5).

Exit 1 when the difference is not rank 1 (sigma ratio < 8) or a layer's direction differs (cosine < 0.999).
"""
import json, struct, sys
import numpy as np


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        t[b] = np.nan if (e == 15 and m == 7) else s * ((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return t


T = e4m3_table()


def deq(wb, sb, shape, sshape):
    w = T[np.frombuffer(wb, np.uint8)].reshape(shape)
    sc = (2.0 ** (np.frombuffer(sb, np.uint8).astype(np.float32) - 127)).reshape(sshape)
    return w * np.repeat(np.repeat(sc, 32, 0), 32, 1)


def main():
    stock, side, out, lam = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    layers = [int(x) for x in sys.argv[5:]]
    f = open(side, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    H = json.loads(f.read(n))
    base = 8 + n

    def sidecar(k):
        a, b = H[k]["data_offsets"]
        f.seek(base + a)
        return f.read(b - a)

    dirs, report, Ws, Ds = [], {"layers": {}}, {}, {}
    for L in layers:
        meta = json.load(open(f"{stock}/L{L}.json"))
        shp, sshp = meta["weight"]["shape"], meta["scale"]["shape"]
        W = deq(open(f"{stock}/L{L}.weight.bin", "rb").read(), open(f"{stock}/L{L}.scale.bin", "rb").read(), shp, sshp)
        Wa = deq(sidecar(f"layers.{L}.attn.wo_b.weight"), sidecar(f"layers.{L}.attn.wo_b.scale"), shp, sshp)
        D = (Wa - W).astype(np.float64)
        ev, V = np.linalg.eigh(D @ D.T)
        dirs.append(V[:, -1])
        report["layers"][L] = {"sigma_ratio": float(np.sqrt(ev[-1] / max(ev[-2], 1e-30))),
                               "rel_change": float(np.linalg.norm(D) / np.linalg.norm(W))}
        Ws[L], Ds[L] = W.astype(np.float64), D
        print(f"   layer {L}: sigma1/sigma2 {report['layers'][L]['sigma_ratio']:.1f}", flush = True)
    M = np.stack(dirs)
    M *= np.sign(M @ M[0])[:, None]
    r = M.mean(0)
    r /= np.linalg.norm(r)
    for L, d in zip(layers, M):
        rW = r @ Ws[L]
        report["layers"][L]["cos_to_mean"] = float(d @ r)
        report["layers"][L]["lam_fit"] = float(-(r @ Ds[L]) @ rW / (rW @ rW))
    with open(out, "wb") as g:
        g.write(b"ABL1" + struct.pack("<II", r.size, len(layers)))
        for L in layers:
            g.write(struct.pack("<If", L, lam) + r.astype("<f4").tobytes())
    report.update(lam = lam, hidden = int(r.size))
    json.dump(report, open(out + ".json", "w"), indent = 1)
    bad = [L for L, x in report["layers"].items() if x["sigma_ratio"] < 8 or x["cos_to_mean"] < 0.999]
    if bad:
        sys.exit(f"!! layers {bad}: the overlay's change is not one shared rank-1 direction there (see {out}.json)")
    print(f"   direction: {len(layers)} layers, cosine to the mean >= {min(x['cos_to_mean'] for x in report['layers'].values()):.4f}")


if __name__ == "__main__":
    main()
