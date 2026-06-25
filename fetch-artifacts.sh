#!/usr/bin/env bash
# fetches the requirement models and data from huggingface
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HF_REPO="${HF_REPO:-mira-thing/mira-voice}"
REV="${REV:-main}"
OUT="$HERE/artifacts"

echo "[fetch] $HF_REPO@$REV -> $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_REPO" --revision "$REV" --local-dir "$OUT" >/dev/null
elif command -v git-lfs >/dev/null 2>&1; then
  tmp="$(mktemp -d)"
  git clone --depth 1 --branch "$REV" "https://huggingface.co/$HF_REPO" "$tmp"
  cp -a "$tmp"/. "$OUT"/ && rm -rf "$OUT/.git" "$tmp"
else
  echo "[fetch] ERROR: need 'huggingface-cli' (pip install huggingface_hub) or git-lfs" >&2
  exit 1
fi

install -Dm0644 "$HERE/models/hey_mira.tflite" "$OUT/models/hey_mira.tflite"

echo "[fetch] done -> $(du -sh "$OUT" | cut -f1)  (c3 wake md5 $(md5sum "$OUT/models/hey_mira.tflite" | cut -c1-8))"
echo "[fetch] note: the go-librespot daemon is NOT in this bundle - build it from ../mira-daemon (bash crosscompile.sh armv6)."
