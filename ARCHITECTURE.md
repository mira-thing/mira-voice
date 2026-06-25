# Mira Voice Stack (`mira-vs`) - architecture

Local-only smart voice on the Car Thing (ARMv8.0-A, no GPU). Nothing leaves the device except the
Spotify calls the daemon already makes.

Pipeline: mic -> oww_wake (wake + endpoint) -> sherpa_asr_server (Zipformer ASR) -> cascade
resolver (Go: phonetic re-rank vs your library) -> play/queue/control, or searchDesktop fallback

The wake/ASR/g2p binaries are aarch64 (run via a bundled ld-linux-aarch64.so.1 loader); the daemon
is armv6. Warm wake->play is ~3.4-3.6s on-device.

## Models
Wake (openWakeWord, 3 tflite stages, runtime oww_wake):
| File | Size | Role |
|---|---|---|
| `melspectrogram.tflite` | 1.1M | audio -> log-mel |
| `embedding_model.tflite` | 1.3M | mel -> 96-d embedding (upstream) |
| `hey_mira.tflite` | ~0.4M | embedding -> "hey mira" score (our c3 model) |

ASR (gigaspeech-Zipformer, runtime `sherpa_asr_server`): encoder 68M + decoder 0.5M + joiner 0.25M +
tokens/bpe. Bundle the `*.int8.onnx` only, never the 260M full-precision encoder.

g2p: `espeak-ng` subprocess, en-us IPA for library names + transcripts.

## Binaries
| Binary | Arch | Source |
|---|---|---|
| `go-librespot` | armv6 | `mira-daemon/` (Go) |
| `oww_wake` | aarch64 | `src/oww_wake.c` |
| `sherpa_asr_server` | aarch64 | `src/sherpa_asr_server.c` |
| `espeak-ng` | aarch64 | debian arm64 .deb |

Shared libs (aarch64, ~40M): the loader, `libtensorflowlite_c.so`, `libonnxruntime.so` +
`libsherpa-onnx-c-api.so`, glibc. Build flags in `BUILD.md`.

## The Go brain (`mira-daemon/daemon/`)
| File | Role |
|---|---|
| `voice.go` | orchestrator: wake -> ASR -> resolve -> Connect command + UI banners |
| `voice_resolver.go` | the cascade: intent grammar + phonetic re-rank + accept floors |
| `voice_g2p.go` | espeak-ng wrapper (memoized/batched IPA) |
| `voice_catalog.go` | Pathfinder catalog sync -> phonetic index (cached, resumable) |
| `voice_sherpa.go` | supervises the sidecar|

