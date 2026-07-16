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
  ( cd "$tmp" && git lfs install --local >/dev/null && git lfs pull )
  cp -a "$tmp"/. "$OUT"/ && rm -rf "$OUT/.git" "$tmp"
else
  echo "[fetch] ERROR: need 'huggingface-cli' (pip install huggingface_hub) or git-lfs" >&2
  exit 1
fi

# verify the downloaded bundle before touching it
if [ -f "$OUT/MANIFEST.txt" ]; then
  echo "[fetch] verifying bundle against MANIFEST.txt..."
  (cd "$OUT" && grep -vE '\./(README\.md|\.gitattributes|THIRD_PARTY_LICENSES)$' MANIFEST.txt > .manifest.artifacts) \
    || { echo "[fetch] ERROR: could not read MANIFEST.txt" >&2; exit 1; }
  (cd "$OUT" && md5sum -c --quiet .manifest.artifacts && rm -f .manifest.artifacts) \
    || { echo "[fetch] ERROR: bundle md5 verification FAILED" >&2
         if head -c 40 "$OUT/models/melspectrogram.tflite" 2>/dev/null | grep -q "git-lfs.github.com"; then
           echo "[fetch] the bundle is git-lfs POINTER files, not the real models." >&2
           echo "[fetch] run 'git lfs install' once, then retry." >&2
         else
           echo "[fetch] (corrupt or partial download)" >&2
         fi
         exit 1; }
else
  echo "[fetch] WARNING: bundle has no MANIFEST.txt; skipping md5 verification" >&2
fi

install -Dm0644 "$HERE/models/hey_mira.tflite" "$OUT/models/hey_mira.tflite"

# the wake model must be the pinned c3 (see the stale-c3 deploy trap)
C3_MD5="36c33c8b53f69ba4b1d935f6d10f9431"
WAKE_MD5="$(md5sum "$OUT/models/hey_mira.tflite" | cut -d' ' -f1)"
[ "$WAKE_MD5" = "$C3_MD5" ] \
  || { echo "[fetch] ERROR: wake model md5 $WAKE_MD5 != pinned c3 $C3_MD5" >&2; exit 1; }

echo "[fetch] done -> $(du -sh "$OUT" | cut -f1)  (c3 wake md5 $(echo "$WAKE_MD5" | cut -c1-8))"
echo "[fetch] note: the go-librespot daemon is NOT in this bundle - build it from ../mira-daemon (bash crosscompile.sh armv6)."
