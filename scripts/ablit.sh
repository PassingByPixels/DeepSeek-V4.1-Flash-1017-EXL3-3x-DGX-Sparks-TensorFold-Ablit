#!/bin/bash
# scripts/ablit.sh: the per-request abliteration (README, "Stock and abliterated on one server"). Run by start.sh
# before the ranks launch, and at the end of scripts/prepare.sh step 2. The ranks must be stopped.
#
# ABLIT=0 (default): the recipe's own binaries on all three ranks; every request is stock.
# ABLIT=1:
#   1. the direction file, once: drowzeys' FP8 wo_b sidecar at ABLIT_REV (a gated repo: accept its terms on the Hub
#      with the account of your HF token), the stock FP8 wo_b of ABLIT_LAYERS by range requests, and
#      tools/ablit_direction.py in ROPE_IMAGE -> $DATA_DIR/ablit/abl-L<layers>-l<lambda>.bin;
#   2. LakyFx's runtime built with ablit/ablit-runtime.patch into $DATA_DIR/build-ablit (once a patch version);
#   3. those binaries in rt/bin and the direction file in rt/ablit/ on all three ranks.
# write_launch_config (common.sh) then exports TF_ABLIT and TF_ABLIT_DEFAULT to the ranks.
. "$(dirname "$0")/common.sh"
B=$DATA_DIR/build; AB=$DATA_DIR/build-ablit; AD=$DATA_DIR/ablit; PATCH=$ROOT/ablit/ablit-runtime.patch
BINS="native/bin/tensorfold-native bin/tf-dsv41-lanes bin/tf-dsv41-check"

sums() { on "$1" "cd ${RT[$1]}/bin && sha256sum tensorfold-native tf-dsv41-lanes tf-dsv41-check 2>/dev/null | cut -c1-16 | tr '\n' ' '"; }
deploy() {  # deploy DIR WHAT: DIR's binaries on every rank whose binaries differ
  local src=$1 what=$2 want r b
  want=$(cd "$src" && for b in $BINS; do sha256sum < "$b" | cut -c1-16; done | tr '\n' ' ')
  for r in 0 1 2; do
    [ "$(sums $r)" = "$want" ] && continue
    running $r && die "ablit: rank $r is running; stop it first (./stop.sh)"
    if [ $r = 0 ]; then for b in $BINS; do cp -p "$src/$b" "${RT[0]}/bin/"; done
    else for b in $BINS; do copy_to $r "$src/$b" "${RT[$r]}/bin/"; done; fi
    [ "$(sums $r)" = "$want" ] || die "ablit: rank $r binaries differ after the copy"
    note "ablit: rank $r now runs $what"
  done
}

[ -d "$B/built" ] || exit 0              # not set up yet: prepare.sh calls this again after it builds
if [ "${ABLIT:-0}" != 1 ]; then
  deploy "$B/built" "the recipe's own binaries (ABLIT=0: every request is stock)"
  exit 0
fi

# the layers: "18-25" or "18 19 20"; the sidecar covers 10-35
LAYERS=$(echo "$ABLIT_LAYERS" | tr ',' ' ' | sed -E 's/([0-9]+)-([0-9]+)/$(seq \1 \2)/g'); LAYERS=$(eval echo "$LAYERS")
for l in $LAYERS; do [ "$l" -ge 10 ] && [ "$l" -le 35 ] || die "ablit: layer $l is outside the overlay's 10-35"; done
F=$AD/$(ablit_file)

if [ ! -f "$F" ]; then
  mkdir -p "$AD"
  note "ablit: drowzeys' FP8 wo_b sidecar at ${ABLIT_REV:0:8} (gated: accept its terms on the Hub for your HF token)"
  python3 "$TOOLS/hub_fetch.py" "$ABLIT_REPO" "$ABLIT_REV" "$AD/overlay" "$ABLIT_SIDECAR" ABLIT_META.json RESPONSIBLE_USE.md \
    || die "ablit: could not fetch $ABLIT_REPO (has this HF token's account accepted its terms?)"
  note "ablit: the stock FP8 wo_b of layers $(echo $LAYERS | tr ' ' ,) by range requests (~42 MB each)"
  python3 "$TOOLS/ablit_fetch.py" "$BASE_REPO" "$BASE_REV" "$AD/stock" $LAYERS || die "ablit: the stock wo_b download failed"
  note "ablit: working out the direction in $ROPE_IMAGE (about a minute)"
  docker run --rm --network none --user "$(id -u):$(id -g)" -v "$AD:/w" -v "$TOOLS:/t:ro" "$ROPE_IMAGE" \
    python3 /t/ablit_direction.py /w/stock "/w/overlay/$ABLIT_SIDECAR" "/w/$(basename "$F").tmp" "$ABLIT_LAMBDA" $LAYERS \
    > "$AD/direction.log" 2>&1 || die "ablit: $(tail -3 "$AD/direction.log")"
  mv "$AD/$(basename "$F").tmp" "$F"; mv "$AD/$(basename "$F").tmp.json" "$F.json"
  note "ablit: $(tail -1 "$AD/direction.log" | sed 's/^ *//')"
fi

want=$(sha256sum < "$PATCH" | cut -c1-16)
have_bins() { local b; for b in $BINS; do [ -x "$AB/built/$b" ] || return 1; done; }
if [ "$(cat "$AB/.patch" 2>/dev/null)" != "$want" ] || ! have_bins; then
  note "ablit: building the runtime with ablit/ablit-runtime.patch ($want, about a minute)"
  rm -rf "$AB"; mkdir -p "$AB"; cp -a "$B/source" "$AB/source"; rm -rf "$AB/source/.zig-cache"
  (cd "$AB/source" && git apply "$PATCH") || die "ablit: the patch does not apply to $B/source"
  (cd "$AB/source" && ZIG_GLOBAL_CACHE_DIR="$DATA_DIR/zig-cache" "$DATA_DIR/zig/zig" build native dsv41-tools -j8 \
     -Dtarget=aarch64-linux-gnu -Dfatbins="$B/kit/build-fatbins" --prefix "$AB/built") > "$AB/build.log" 2>&1 \
    || die "ablit: zig build failed: $(tail -5 "$AB/build.log")"
  echo "$want" > "$AB/.patch"
fi
deploy "$AB/built" "the patched binaries (ABLIT=1)"
for r in 0 1 2; do
  on $r "mkdir -p ${RT[$r]}/ablit"
  if [ $r = 0 ]; then cp -p "$F" "${RT[0]}/ablit/"; else copy_to $r "$F" "${RT[$r]}/ablit/"; fi
done
note "ablit: patch $want; direction $(basename "$F"); $SERVED_NAME = $([ "$ABLIT_DEFAULT" = 1 ] && echo abliterated || echo stock); $SERVED_NAME-ablit / -stock choose per request"
