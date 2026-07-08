# Building the artifacts

`collect-artifacts.sh` gathers prebuilt outputs into `artifacts/`. Rebuild a binary only when its
source changes, then rerun it. All aarch64 bins target the A53:
`-march=armv8-a+crypto+crc -mtune=cortex-a53 -moutline-atomics`.

`src/` holds the two project-owned C sources:
- `oww_wake.c` - wake-word listener + command capture (TFLite C API + tinyalsa)
- `sherpa_asr_server.c` - persistent ASR sidecar (sherpa-onnx C API)

Both build with the toolchain image from the vendored `Dockerfile` (see below).

| Artifact | Build |
|---|---|
| `go-librespot` (armv6) | `cd mira-daemon && bash crosscompile.sh armv6` |
| `oww_wake` (aarch64) | aarch64 gcc, link the TFLite C API + static tinyalsa |
| `sherpa_asr_server` (aarch64) | aarch64 gcc, link `-lsherpa-onnx-c-api -lonnxruntime` |
| `espeak-ng` (aarch64) | extract from the debian arm64 `.deb` |

Build the toolchain image once from the repo root: `docker build -t oww-wake-aarch64 .` (it builds the
TFLite runtime + `oww_wake`). `sherpa_asr_server` reuses the same image as the aarch64 toolchain.

Models (no build): wake `c3.tflite` + melspectrogram/embedding (openWakeWord); ASR
gigaspeech-Zipformer int8 ONNX (k2-fsa release); espeak-ng-data (trimmed to en-us).
