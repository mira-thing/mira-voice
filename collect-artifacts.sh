#!/usr/bin/env bash
# assemble the deployable voice bundle 
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${1:-$HERE/artifacts}"

VS="$ROOT/voice-stack"
WW="$ROOT/wakeword-experiment"
SHERPA_DIR="$VS/moonshine/zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12"

say() { printf '\033[0;36m[collect]\033[0m %s\n' "$*"; }
die() { printf '\033[0;31m[collect] MISSING: %s\033[0m\n' "$*" >&2; exit 1; }
need() { [ -e "$1" ] || die "$1"; }

need "$WW/output/oww_wake"
need "$VS/moonshine/sidecar/sherpa_asr_server"
need "$VS/espeak-build/output/espeak-ng"
need "$WW/pinned_best/c3.tflite"
need "$WW/output/models/melspectrogram.tflite"
need "$WW/output/models/embedding_model.tflite"
need "$SHERPA_DIR/encoder-epoch-30-avg-1.int8.onnx"
need "$SHERPA_DIR/decoder-epoch-30-avg-1.int8.onnx"
need "$SHERPA_DIR/joiner-epoch-30-avg-1.int8.onnx"
need "$SHERPA_DIR/tokens.txt"
need "$SHERPA_DIR/bpe.model"
# tokens.txt must be a real vocab, not a truncated download (gigaspeech bpe = 500 lines)
tok_lines=$(wc -l < "$SHERPA_DIR/tokens.txt")
[ "$tok_lines" -ge 100 ] || die "tokens.txt suspiciously short ($tok_lines lines)"
need "$VS/espeak-build/output/espeak-ng-data"
need "$ROOT/mira-daemon/go-librespot-armv6"

say "output -> $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"/bin "$OUT"/lib "$OUT"/models "$OUT"/espeak-ng-data "$OUT"/daemon \
         "$OUT/zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12"

install -m0755 "$WW/output/oww_wake"                    "$OUT/bin/oww_wake"
install -m0755 "$VS/moonshine/sidecar/sherpa_asr_server" "$OUT/bin/sherpa_asr_server"
install -m0755 "$VS/espeak-build/output/espeak-ng"      "$OUT/bin/espeak-ng"

cp -a "$WW/output/lib/."                "$OUT/lib/"
cp -a "$VS/espeak-build/output/lib/."   "$OUT/lib/"
cp -a "$VS/moonshine/sidecar/lib/."     "$OUT/lib/"

say "stripping prebuilt runtimes (debug info only)"
if command -v aarch64-linux-gnu-strip >/dev/null 2>&1; then
  ( cd "$OUT/lib" && aarch64-linux-gnu-strip --strip-unneeded \
      libonnxruntime.so libsherpa-onnx-c-api.so libtensorflowlite_c.so )
else
  docker run --rm -v "$OUT/lib":/s -w /s oww-wake-aarch64 sh -c \
    'for l in libonnxruntime.so libsherpa-onnx-c-api.so libtensorflowlite_c.so; do \
       aarch64-linux-gnu-strip --strip-unneeded "$l"; done'
fi

say "dropping espeak audio chain (g2p runs -q --ipa, no synthesis)"
if command -v patchelf >/dev/null 2>&1; then
  patchelf --remove-needed libpcaudio.so.0 --remove-needed libsonic.so.0 "$OUT/lib/libespeak-ng.so.1"
else
  docker run --rm -v "$OUT/lib":/s -w /s firmware-builder:latest \
    patchelf --remove-needed libpcaudio.so.0 --remove-needed libsonic.so.0 libespeak-ng.so.1
fi

say "pruning lib/ to the runtime DT_NEEDED closure"
_keep=$(mktemp); printf 'ld-linux-aarch64.so.1\n' > "$_keep"
_q=$(for b in "$OUT"/bin/oww_wake "$OUT"/bin/sherpa_asr_server "$OUT"/bin/espeak-ng; do
       readelf -d "$b" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'; done)
while [ -n "$_q" ]; do
  _nx=""
  for _l in $_q; do
    grep -qxF "$_l" "$_keep" && continue
    printf '%s\n' "$_l" >> "$_keep"
    [ -f "$OUT/lib/$_l" ] && _nx="$_nx $(readelf -d "$OUT/lib/$_l" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')"
  done
  _q=$_nx
done
for _f in "$OUT"/lib/*; do
  grep -qxF "$(basename "$_f")" "$_keep" || rm -f "$_f"
done
rm -f "$_keep"


cp -a "$WW/output/models/melspectrogram.tflite" "$OUT/models/"
cp -a "$WW/output/models/embedding_model.tflite" "$OUT/models/"
cp -a "$WW/pinned_best/c3.tflite"               "$OUT/models/hey_mira.tflite"

sz="$OUT/zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12"
cp -a "$SHERPA_DIR/encoder-epoch-30-avg-1.int8.onnx" "$sz/"
cp -a "$SHERPA_DIR/decoder-epoch-30-avg-1.int8.onnx" "$sz/"
cp -a "$SHERPA_DIR/joiner-epoch-30-avg-1.int8.onnx"  "$sz/"
cp -a "$SHERPA_DIR/tokens.txt"  "$sz/"
cp -a "$SHERPA_DIR/bpe.model"   "$sz/"

ED="$VS/espeak-build/output/espeak-ng-data"
for f in phondata phonindex phontab intonations en_dict; do cp -a "$ED/$f" "$OUT/espeak-ng-data/"; done
cp -a "$ED/lang" "$ED/voices" "$OUT/espeak-ng-data/"
install -m0755 "$ROOT/mira-daemon/go-librespot-armv6" "$OUT/daemon/go-librespot-armv6"

( cd "$OUT" && find . -type f ! -name MANIFEST.txt -exec md5sum {} \; | sort -k2 > MANIFEST.txt )
TOTAL=$(du -sh "$OUT" | cut -f1)
C3_MD5="36c33c8b53f69ba4b1d935f6d10f9431"
WAKE_MD5=$(md5sum "$OUT/models/hey_mira.tflite" | cut -d' ' -f1)
[ "$WAKE_MD5" = "$C3_MD5" ] \
  || { printf '\033[0;31m[collect] FATAL: wake model md5 %s != pinned c3 %s (stale-c3 trap)\033[0m\n' "$WAKE_MD5" "$C3_MD5" >&2; exit 1; }
say "wake model md5: $WAKE_MD5 (pinned c3 OK)"
say "bundle assembled: $TOTAL across $(grep -c . "$OUT/MANIFEST.txt") files"
say "DONE -> $OUT"
