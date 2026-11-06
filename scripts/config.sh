# Pins and defaults. Put your own settings in scripts/local.sh (read after this file), not here.

# The checkpoint this recipe serves, and the original weights its Engram tables come from
QUANT_REPO=${QUANT_REPO:-easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw}
QUANT_REV=${QUANT_REV:-30efbfcc58fc105588d480a40989dac3b0ae4146}
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_FILES="model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json"
ENGRAM_SHA256_47=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
ENGRAM_SHA256_48=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd

# Runtime: LakyFx's TensorFold TP3 port for three DGX Sparks, at a pinned tag. Its extractor checks the frozen
# TensorFold native runtime snapshot below; Zig builds it for GB10.
RUNTIME_GIT=https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark.git
RUNTIME_TAG=v1.2.2TF
RUNTIME_REV=14a6d49c7cb99b8ce25f9755541da7e7d57cb1d1
RUNTIME_SNAPSHOT=61357debe730ac9ffe20e760edcddbc253b0e525
ZIG_URL=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8

# NCCL 2.30.7 with the switchless patches (direct ConnectX-7 cables, no switch): alexellis/switchless-nccl v0.0.1
NCCL_URL=https://github.com/alexellis/switchless-nccl/releases/download/v0.0.1/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz
NCCL_TGZ_SHA256=b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f
NCCL_LIB_SHA256=78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57

# RoPE tables: the runtime's make_rope.py on CPU PyTorch in this image (no network), at its 10M reference pool (it
# checks LakyFx's qualified hashes), then cut to the rows POOL needs. The 4M cut must match these.
ROPE_IMAGE=${ROPE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
ROPE_REF_POOL=10000000
ROPE_4M_ROWS=4001792
ROPE_4M_SHA256="rope-plain-cos.f32=8fbb1d468fb70330999c0fe5e94b7300b05abcbb87fa7489db68fa5eca1ac515
rope-plain-sin.f32=f1b678d1261a576f284b62175b61270be5295401eee6d09c7ad450eb80cae930
rope-compressed-cos.f32=722ede80087106d4301300fbae5109dd2c8f6f02ae363756c7eaadb7321526b7
rope-compressed-sin.f32=982c40beb3d55236e6f51de1cdbf2c0f36aa8e8bcabfef4f06991176e5dc91cf"

# The 1017 chat template the launcher requires (LakyFx's qualified one; identical to the checkpoint's)
CHAT_TEMPLATE_SHA256=7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e

# Expected outputs of a clean setup, recorded from our end-to-end run on 2026-10-21 (the setup prints them; set either to
# empty to skip the check)
PLAN_SHA256=${PLAN_SHA256-926e10858b41a0d11c94c4a408cafa3e5fdb44dc212c9a8756054a3713919261}
RANK_SHA256=${RANK_SHA256-11264938fbfe2daa3c316cb6c3ec2695b829fca7d79f7ccd5ff7646059ffe917 7e4b1c4f2452728026cbec03f3777c288d51a150d89c5efbdcc8584615460d73 af38459f37b3cb7c7adf229dff00c19839e23372a2f3c3e6a90706826a4ad811}   # the three rank caches

# Per-request abliteration (README, "Stock and abliterated on one server"). ABLIT=1 serves both on one server:
# the -ablit and -stock model ids choose per request, ABLIT_DEFAULT what the plain SERVED_NAME gives. The edit is
# drowzeys' Keys direction on attn.wo_b, applied at run time; the overlay repository is gated (accept its terms).
ABLIT=${ABLIT:-0}
ABLIT_DEFAULT=${ABLIT_DEFAULT:-1}     # 1: SERVED_NAME is abliterated; 0: SERVED_NAME is stock
ABLIT_LAYERS=${ABLIT_LAYERS:-18-25}   # layers the edit applies to (the overlay covers 10-35; 18-25 measured best)
ABLIT_LAMBDA=${ABLIT_LAMBDA:-3.5}     # edit strength (the overlay's)
ABLIT_REPO=drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
ABLIT_REV=de412d2249632903dace23399526e00a145aada5
ABLIT_SIDECAR=wo_b_l10_35_1017.safetensors
# With ABLIT=1 (the patched runtime): a decode round of the other streams between a long prompt's chunks, so a prompt
# being read does not stop everyone else (TF_DS_PREFILL_YIELD, ported from Mia's AI Lab 0023). 0: whole prompt first.
PREFILL_YIELD=${PREFILL_YIELD:-1}

# NCCL on a switchless triangle: the switchless build's ring mode (README, Requirements). RANK_ENV: anything else to
# export to every rank (e.g. NCCL_DEBUG=INFO).
NCCL_ENV=${NCCL_ENV-NCCL_SWITCHLESS_RING_ONLY=1 NCCL_ALGO=Ring}
RANK_ENV=${RANK_ENV:-}

# Where everything lives on each Spark. Head: ~700 GB at the peak of the setup (checkpoint ~253, Engram source ~203,
# three rank caches ~86 each before ranks 1 and 2 move to their Sparks, Engram cut ~68); ~400 GB after. Others ~160 GB.
DATA_DIR=${DATA_DIR:-$HOME/.cache/dsv41-1017-tp3}
ARM=TP3A                          # the runtime's name for this checkpoint's rank caches and profile

# The Sparks (see README, Requirements). WORKER1 and WORKER2: ssh targets of ranks 1 and 2, key-based from the head.
WORKER1=${WORKER1:-}
WORKER2=${WORKER2:-}
MASTER_PORT=29660                 # the runtime's rank control ports: 29660 and 29661 on the head

# Server
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}
SERVED_NAME=${SERVED_NAME:-deepseek-v4.1-flash}
POOL=${POOL:-4000000}             # tokens of KV shared by all requests
CONTEXT=${CONTEXT:-401408}        # tokens per request (prompt + output); may exceed POOL / PARALLEL
PARALLEL=${PARALLEL:-12}          # concurrent requests
PROMPT_CACHE_GIB=${PROMPT_CACHE_GIB:-2}
REASONING=${REASONING:-}          # empty: the server's default; or max, high, medium, low, none
API_KEY_FILE=${API_KEY_FILE:-}    # a file holding the API key; every route but /health then needs it
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-100}   # warn below this much MemAvailable on a Spark before loading
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}   # the CUDA runtime libraries the binaries load (DGX OS default)
ALLOW_4K_PAGES=${ALLOW_4K_PAGES:-0}   # we qualified on the 64K-page kernel only
