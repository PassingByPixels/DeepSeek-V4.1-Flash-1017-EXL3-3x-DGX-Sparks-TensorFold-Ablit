# DeepSeek-V4.1-Flash-1017 EXL3 3.5 bpw on 3× DGX Spark with TensorFold — stock and abliterated on one server

> **This is a fork** of [easthambuilds' 3-Spark recipe](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold)
> (`7d72d1e`). It adds one thing: **stock and abliterated replies from the same server, chosen per request**
> (`ABLIT=1`, [below](#stock-and-abliterated-on-one-server)), with
> [drowzeys' Keys abliteration](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) applied at run time
> instead of baked into the weights. It also carries two setup fixes for four-rail triangles (upstream issues
> [#1](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold/issues/1) and
> [#2](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold/issues/2)) and a readahead
> check. With `ABLIT=0` (the default) it serves exactly what the donor recipe serves. The rest of this README is the
> donor's, with these changes marked.

Serve [**DeepSeek-V4.1-Flash-1017**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) from three NVIDIA DGX
Sparks (GB10, 128 GB each, cabled to each other by their ConnectX-7 ports) through an OpenAI-compatible API, with
**12 concurrent requests** over a **4M-token shared KV pool**, images, and DSpark drafting.

This repository is a set of scripts and pins, not an engine. The runtime is **LakyFx's TensorFold TP3 port**,
[`deepseek-v41-flash-exl3-tp3-spark`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark), built from its
pinned sources with its kernels and engine unchanged. It is a three-rank port of Bertholomus AI's DeepSeek Zig engine
for [TensorFold](https://github.com/ashhart/TensorFold). What is ours:

- **The checkpoint:** [`easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw`](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw),
  an EXL3 quant of the 1017 weights with agentic calibration, made with published provenance
  ([model card](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)).
- **The fit:** a checkpoint plan for the runtime's layout (`tools/make_plan.py`), one exact BF16 → F32 conversion in
  its rank-cache exporter, and a launcher that takes a triangle of direct cables (`tools/patch_runtime.py`).
- **The setup:** `start.sh` / `scripts/prepare.sh` find the cabling, build, download and check every byte, write each
  rank's cache and Engram cut on the head and move them to the other two Sparks, then serve and smoke-test.

| | |
| --- | --- |
| Checkpoint | [`easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw`](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) @ `30efbfcc`, ~253 GB, plus DeepSeek's original shards 47–48 (Engram tables, ~203 GB) |
| Runtime | [`LakyFx/deepseek-v41-flash-exl3-tp3-spark`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark) tag `v1.2.2TF` (`14a6d49`), TensorFold runtime snapshot `61357deb`, built with Zig 0.17.0 for sm_121 |
| NCCL | 2.30.7 with the switchless patches, [`alexellis/switchless-nccl`](https://github.com/alexellis/switchless-nccl) `v0.0.1`, sha256-checked |
| RoPE tables | the runtime's `make_rope.py` on CPU in `nvcr.io/nvidia/pytorch:26.07-py3`, cut to the pool |
| API | `http://127.0.0.1:8888/v1`, model id `deepseek-v4.1-flash` |
| Defaults | pool 4,000,000 tokens, 401,408 tokens per request, 12 streams, 2 GiB prompt cache, the server's default reasoning effort |

## Stock and abliterated on one server

With `ABLIT=1` in `scripts/local.sh`, one server answers both ways, request by request:

| Request | Reply |
|---|---|
| `"model": "<SERVED_NAME>-stock"` | stock |
| `"model": "<SERVED_NAME>-ablit"` | abliterated |
| `"model": "<SERVED_NAME>"` | `ABLIT_DEFAULT`: 1 abliterated (default), 0 stock |
| any of these plus `"ablit": true` or `false` | that mode |

`/v1/models` lists all three ids. Two clients can use different modes at the same time: the requests share the
batch, and each row gets its own mode. An abliterated request never reuses a stock request's cached prompt (its KV
differs from layer 18 on), and the other way round.

**Before you set `ABLIT=1`:** the overlay repository is gated. Accept its terms on the Hub
([drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated),
including its Responsible Use agreement) with the account of the HF token on the head. The recipe downloads it with
that token; it does not ship any part of it.

### How it works

drowzeys' overlay changes `attn.wo_b` of layers 10–35 as `W' = W − λ r rᵀ W`: one refusal direction `r` in the
5120-wide hidden state, λ = 3.5. Because the change is rank 1, the runtime can make the same change to `wo_b`'s
output instead of to its weights: `y ← y − λ · mask[row] · (r · y) r`, with `mask[row]` 1 on the rows of an abliterated
request and 0 on a stock one. Each rank applies it to its own fp32 partial before the all-gather; the edit is linear,
so the sum gets the same edit. It costs three small cuBLAS calls a layer, inside the CUDA graphs.

`scripts/ablit.sh` (run by `start.sh` and `prepare.sh`, ranks stopped) does the setup once:

1. downloads the overlay's FP8 sidecar `wo_b_l10_35_1017.safetensors` at a pinned revision (`hub_fetch.py`, sha256
   checked) and the stock FP8 `wo_b` of `ABLIT_LAYERS` from DeepSeek's checkpoint by range requests (~42 MB a layer,
   `tools/ablit_fetch.py`);
2. works out `r` in the RoPE step's PyTorch image (`tools/ablit_direction.py`, no network): the top singular vector of
   each layer's difference, averaged. It stops unless every difference is rank 1 (σ1/σ2 ≥ 8; measured 14–17) and
   every layer gives the same direction (cosine ≥ 0.999; measured ≥ 0.9998);
3. builds the runtime with `ablit/ablit-runtime.patch` (Apache-2.0, like the TensorFold sources it changes; summary at
   its top) into `$DATA_DIR/build-ablit`, and puts those binaries and the direction file on all three ranks.

`ABLIT=0` puts the recipe's own binaries back. Settings: `ABLIT`, `ABLIT_DEFAULT`, `ABLIT_LAYERS` (default `18-25`;
the overlay covers 10–35), `ABLIT_LAMBDA` (default 3.5). A new layer set or λ is a new direction file and a
restart, no re-export of the rank caches.

### A long prompt no longer stops the others (`PREFILL_YIELD`, on with `ABLIT=1`)

The served runtime reads a new prompt to the end before any other stream decodes again. With the patched runtime
(`ABLIT=1`) the fork also carries Mia's AI Lab's prefill yield (their patch `0023`, ported): a decode round of the
live streams between the prompt's 2,048-token chunks. Measured here, three streams decoding while a 102,939-token
prompt is read:

| `PREFILL_YIELD` | longest pause of the other streams | the prompt's time to its reply |
|---|---|---|
| `0` (served behaviour) | 56.3 s | 56.6 s |
| `1` (default with `ABLIT=1`) | 2.8 s | 67.3 s (+19 %) |

Same reply either way. With it, the host also reads prompts one chunk per turn, rotating between them (Mia's AI Lab
`0031` by Mauvis Ledford, and `0032`, ported), so a request that arrives during a long prompt starts at once:

| A ~100k-token prompt is being read; 5 s later | before | now |
|---|---|---|
| a short request's reply | 57.6 s | 0.6 s |
| an image request's reply | (same wait) | 2.2 s, correct |
| the long prompt's own reply | 62.6 s | 62.4 s |

`PREFILL_YIELD=0` in `scripts/local.sh` turns both off.

### Measured (our three Sparks, 4K-page kernels, `ALLOW_4K_PAGES=1`, GPU clocks capped at 2,000 MHz)

| | stock | abliterated, run-time L18-25 |
|---|---|---|
| Refusals, 24 dual-use prompts (`ablit/eval/gate.py`) | 18/24 | 0/24 |
| Refusals, 36 held-out prompts (`refuse2.py`) | 27/36 | 0/36 |
| MMLU-Pro, 240 questions, thinking off (`mmlu.py`) | 0.787 | 0.796 |
| GSM8K, 120 questions (`eval2.py`) | 116 | 117 |
| Decode, 1 / 4 / 8 streams (tok/s) | 70.0 / 135.1 / 175.2 | 70.5 / 136.5 / 175.6 |

- **Stock mode is unchanged by the patch:** 24/24 greedy replies byte-identical to the unpatched engine.
- **Mixed batches:** stock and abliterated requests at the same time give each mode's own refusal counts (18/24 and
  0/24); greedy text matched the solo runs on 23-24 of 24 prompts (batched decode is not bit-exact, in stock mode too).
- **Prompt cache:** stock, ablit, ablit, stock on one 7.8K-token prefix gave cached tokens 0, 0, 7,831, 7,831.
- **Why layers 18–25:** we baked layer subsets of the overlay into rank caches and measured each. The full overlay
  (10–35) costs capability (MMLU-Pro 0.750: paired against stock it lost 15 questions and gained 6). Every subset that
  starts at layer 18 lost 5 and gained 8. 18–25 is the smallest that keeps 0/60 refusals; 18–23 and 22–27 let one
  prompt refuse again.
- **What was tested:** on our three Sparks, from an already set-up `DATA_DIR`: `scripts/ablit.sh` from nothing
  (download, direction, build, deploy), `ABLIT=0` and `ABLIT=1` starts, the dry-run preflight. A first-time setup of
  new Sparks with this fork has not been run end to end.
- The refusal sets detect a bypass (a regex on the first 400 characters of the reply); they are not a safety benchmark.
- Only the OpenAI routes read the mode (`/v1/chat/completions`, tested; `/v1/completions` shares its code). The
  Anthropic and Responses routes do not, so they probably give stock replies (untested).

## Results

Measured on our three GB10 (Linux 7.0 64K-page kernel, driver 580.178.04) with the same runtime build, checkpoint
weights, runtime changes and settings that this recipe sets up, which we first set up by hand.
**GPU clocks capped at 2,000 MHz on all three** for every number here. Raw results are in the model's
`provenance/logs/tp3a/`.

**Status:** end-to-end run passed on our three Sparks on 2026-10-21. With the checkpoint and Engram shards already on
disk, `./start.sh` checked them against the Hub, made the plan, the Engram cuts and the three rank caches, and set up all
three Sparks in 41 minutes, then loaded in 62 s and answered the smoke test. The runtime build and RoPE tables (about two
minutes) had run earlier that morning, in a first run that stopped on a bug since fixed. Its three rank caches are
byte-identical to the hand-built ones the numbers above were measured on; their sha256 are pinned in `scripts/config.sh`.

- **Decode, one stream** (Bertholomus AI's `kit_bench`, medians, MTP drafts on): 109.2 tok/s writing code and 71.2
  writing prose (set a). Our 2.9 bpw quant on two Sparks (TP2) at the same cap: 85.7 and 56.1.
- **Concurrent streams, total:** 2 streams 99–106, 4 streams 133–144, 8 streams 169–181 tok/s.
- **Prefill:** 2,223 / 2,269 / 2,115 tok/s at 8K / 32K / 128K-token prompts.
- **Trade-off:** on this same runtime, our 2.9 bpw weights decode 8–10% faster on code; prose is level. This
  checkpoint spends that on fidelity: KL against DeepSeek's FP8 release is 15–39% lower than the 2.9 bpw quants'
  on five test sets ([model card](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)).
- **Benchmarks:** GSM8K 97.0, MMLU 88.4, BFCL v3 81.5 (thinking off), needle 12/12 at 8K–250K. Level with our
  2.9 bpw quant; no paired difference is significant.
- **Memory:** the model takes 92.5 GiB on the head. With the 4M pool allocated and idle, the three Sparks have
  17.8 / 19.3 / 19.2 GiB available. Under a full-pool stress test (ten requests of 389,900 tokens, all completed) the
  lowest was 9.9 GB on the head and 19.5 GB on the other two. A smaller `POOL` or `PROMPT_CACHE_GIB` buys headroom.

## Requirements

- **Three DGX Sparks** (or GB10 systems with 128 GB unified memory) with nothing else large on their GPUs. Each
  wants ~100 GiB available when its rank starts; `start.sh` first drops the runtime's own files from the page cache
  (`tools/evict_page_cache.py`): GB10 reclaims page cache too slowly for CUDA.
- **Cabled as a triangle:** each Spark's two ConnectX-7 ports go to the other two, three cables in all, no switch.
  Every cable shows up as two RoCE devices per end (one per PCIe function, e.g. `rocep1s0f1` / `roceP2p1s0f1`), and
  each of those needs an IPv4 address, in a subnet of its own shared only with the device at the cable's other end:
  six subnets in all. The preflight works out which port reaches which Spark, the RoCE v2 GID index and the rest
  from these addresses (`DRY_RUN=1 ./start.sh` shows it).
- **Host routes:** with one subnet per cable, the third Spark cannot reach the head's cable to the second, and the
  two workers cannot reach each other's NCCL socket address. The preflight lists the three /32 routes this needs,
  each over the cable the two Sparks share, e.g. `sudo ip route replace 192.168.200.2/32 via 192.168.210.2 dev
  enp1s0f1np1`. `APPLY_ROUTES=1` adds them with sudo until the next reboot; make them permanent in your netplan or
  NetworkManager profiles.
- **The 64K-page kernel** (`linux-nvidia-64k`, `getconf PAGESIZE` = 65536) on all three: we measured on it only.
  `ALLOW_4K_PAGES=1` skips the check. Memlock unlimited (`ulimit -l`) for RDMA.
- **Key-based ssh** from the head (rank 0, which runs the API) to the other two (`WORKER1`, `WORKER2`).
- `python3`, `rsync` and GNU `tar` on all three; Docker (for the RoPE step only), `git` and `curl` on the head. The
  CUDA runtime libraries in `/usr/local/cuda/lib64` (`CUDA_LIB`), as on DGX OS.
- **Disk:** on the head, ~720 GB under `DATA_DIR` at the peak of the setup (checkpoint ~253 GB, Engram source
  ~203 GB, three rank caches ~86 GB each before two of them move, Engram cut ~68 GB, kit and RoPE ~8 GB); ~400 GB
  after. On the other two, ~160 GB each.
- **Readahead (this fork):** the runtime loads its rank cache with buffered reads, which the default 128 KB
  readahead caps at ~1 GB/s. With 4 MB the load took 79 s instead of 140 s on our Sparks (one of them lost the most at
  128 KB). `start.sh` warns below 1 MB. To set it on each Spark, until reboot and then permanent:
  `echo 4096 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb`, and a udev rule
  `ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/read_ahead_kb}="4096"` in
  `/etc/udev/rules.d/60-nvme-readahead.rules`.
- **NCCL on four rails (this fork):** with both halves of each cable addressed, NCCL gets one HCA per peer
  (`tools/topology.py`, [#2](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold/issues/2)),
  and every rank runs with `NCCL_SWITCHLESS_RING_ONLY=1 NCCL_ALGO=Ring` (`NCCL_ENV`; more in `RANK_ENV`).
- **Same GPU clock cap on all three** if you compare numbers: the ranks wait for each other every step, so the
  slowest clock sets the pace. We use `sudo nvidia-smi -lgc 0,2000`.

## Quick start

On the head:

```bash
git clone https://github.com/PassingByPixels/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold-Ablit.git
cd DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold-Ablit
cp scripts/local.sh.example scripts/local.sh   # WORKER1, WORKER2
DRY_RUN=1 ./start.sh                            # checks the Sparks, cables and routes; changes nothing
./start.sh
```

The first run sets up all three Sparks (below), which takes hours, mostly downloading. Later starts only load
(a few minutes). `./start.sh` ends with one request through all three ranks and prints the endpoint:

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 4000
}'

./start.sh restart        # restart all three ranks (picks up changed settings)
./stop.sh                 # stop all three; each rank's log is kept in rt/logs/archive
./status.sh               # processes, memory, GPUs and the API's /health
```

The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`. The server listens on
`127.0.0.1` by default. Set `HOST=0.0.0.0` in `scripts/local.sh`, together with `API_KEY_FILE`, to serve the
network. `GET /metrics` gives Prometheus counters; `GET /health` a live view.

## What `scripts/prepare.sh` does

Run on the head; it drives the other two over ssh. Each step skips work already done, so it is safe to re-run.

1. **Preflight and topology:** the GPU, page size, memlock, tools and disk on each Spark; the cables
   (`tools/node_facts.py`, `tools/topology.py`): four rails per Spark, which rails reach which peer, the GID index,
   the master address, the NCCL address range; the host routes; the workers reach the master over the cables.
2. **Runtime:** LakyFx's repository at the pinned commit. Its `extract.py` checks and unpacks the frozen TensorFold
   sources and kernel kit. Zig 0.17.0 (sha256-checked) builds the head and follower binaries. NCCL 2.30.7 switchless
   (sha256-checked, archive and library). `tools/patch_runtime.py` writes the patched exporter and launcher (below).
   Binaries and NCCL are copied to the other two and compared, and `ldd` must resolve every library.
3. **Kit and RoPE tables:** the runtime's `make_rope.py` at its 10M reference pool, on CPU without network; it
   checks LakyFx's qualified hashes. `tools/rope_cut.py` keeps the rows the pool needs (for the 4M pool, the first
   4,001,792 rows, which must match pinned hashes).
4. **Weights:** `tools/hub_fetch.py` downloads the checkpoint and DeepSeek's shards 47–48 at pinned revisions and
   checks every file against the Hub: sha256 for large files, the git blob id for small ones.
5. **Plan, receipt, model files:** `tools/make_plan.py` maps every tensor the runtime loads to a shard and byte range
   of the checkpoint and checks it against the runtime's reference geometry (144,286 keyed tensors, none missing). The
   runtime's `source_verify.py` re-hashes every shard against the Hub's sha256. `tools/make_arm.py` writes the
   runtime's config and the model folder (tokenizer, chat template, the 266 vision tensors compared byte for byte).
6. **Engram:** the runtime's `prepare_engram.py` cuts each rank's 8 of the 24 hash columns (layers 1 and 14) from
   DeepSeek's shards as sparse files; ranks 1 and 2 get theirs over the link, checked by sha256 there.
7. **Rank caches:** the runtime's `rank_cache.py` writes each rank's weights (~86 GB) on the head, and `rank_verify.py`
   re-reads them in full. Ranks 1 and 2 get theirs over the link; each must match its exporter receipt.
8. **Launcher:** the cluster config (from step 1), the profile and `run_rank.sh` on all three, then the launcher's own
   checks for each rank.

**The two runtime changes,** both written to new files by `tools/patch_runtime.py`, which stops if any edit does not
match its pinned source exactly once:

- The exporter and its verifier gain one exact conversion, BF16 → F32 (the F32 bit pattern is the BF16 one shifted up
  16 bits). This checkpoint keeps the source's BF16 Engram k/q projections of layers 1 and 14; LakyFx's own checkpoint
  stores them as F32, and the runtime reads them as F32 either way.
- The launcher takes per-rank NCCL socket interfaces, HCAs and library paths (a triangle with a subnet per cable has
  no shared management interface), any profile, the served name and an API key file, and `--check`.

## Configuration

`scripts/config.sh` holds the pins and defaults. Override them in `scripts/local.sh`: `WORKER1`, `WORKER2`,
`DATA_DIR`, `HOST`, `PORT`, `API_KEY_FILE`, `SERVED_NAME`, `POOL`, `CONTEXT`, `PARALLEL`, `PROMPT_CACHE_GIB`,
`REASONING`, `FABRIC_PEER1` / `FABRIC_PEER2` (copy the setup's files over these cable addresses), `APPLY_ROUTES`,
`MIN_AVAIL_GIB`, `CUDA_LIB` and `ALLOW_4K_PAGES`. Serving settings take effect at `./start.sh restart`; a larger
`POOL` re-cuts the RoPE tables first.

**Context:** `CONTEXT` is each request's window and `POOL` the KV all requests share. A request's window may exceed
`POOL / PARALLEL`; requests then compete for the pool. The runtime and its qualification use 401,408 tokens per request.

## Credits

- **[easthambuilds](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold):** the
  recipe this fork is built on: the checkpoint, the plan, the setup and everything not listed under this fork's changes.
- **[drowzeys](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated):** the Keys abliteration
  of DeepSeek-V4.1-Flash-1017 (MIT, gated with a Responsible Use agreement) whose direction the run-time edit applies.
  Used with drowzeys' agreement; the recipe fetches it with your own accepted token.

- **[DeepSeek](https://huggingface.co/deepseek-ai):** DeepSeek-V4.1-Flash-1017 (MIT).
- **[LakyFx](https://github.com/LakyFx):** the TensorFold TP3 runtime for three DGX Sparks (MIT): its launcher,
  exporter, Engram and RoPE tools, profiles and kernel kit, used as published apart from the two changes above; and the
  per-layer routed-expert widths of their 1017 export, which the checkpoint follows.
- **[Ashley Hart](https://github.com/ashhart) and the [TensorFold](https://github.com/ashhart/TensorFold)
  contributors:** the native Zig CUDA engine. **[Bertholomus AI](https://github.com/bertholomus):** the DeepSeek
  V4.1 Zig engine family the TP3 port builds on, and `kit_bench`.
- **[Mia-AiLab](https://huggingface.co/Mia-AiLab):** the 2.9 bpw per-tensor map the checkpoint's non-expert tensors
  follow, the chat template, and the conventions of their DGX Spark recipes.
- **Pollard precision map:** [bot-lab-21](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)
  and [WestWaters](https://github.com/WestWaters/pollard-weights).
- **[turboderp](https://github.com/turboderp-org/exllamav3):** ExLlamaV3 and the EXL3 format.
- **[Alex Ellis / OpenFaaS](https://github.com/alexellis/switchless-nccl)** and
  **[SparkRing](https://github.com/FujitsuPolycom/sparkring):** the switchless NCCL 2.30.7 build and patches, on
  [NVIDIA NCCL](https://github.com/NVIDIA/nccl).

Not affiliated with or endorsed by DeepSeek, NVIDIA, LakyFx, the TensorFold authors, Bertholomus AI or Mia-AiLab.

## License

Apache-2.0 for the scripts and tools in this repository (`LICENSE`), including this fork's changes
(`scripts/ablit.sh`, `tools/ablit_*.py`, `ablit/`, and the edits listed at the top). `ablit/ablit-runtime.patch` changes
Apache-2.0 TensorFold sources and is Apache-2.0. The abliteration overlay keeps its own license and terms. The checkpoint, runtime, NCCL build and
DeepSeek's weights keep their own licenses.
