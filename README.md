# Mira Voice Stack

On-device smart voice for the Spotify Car Thing: wake ("hey mira") -> ASR ->
cascade re-rank against your library -> play/control.
100% local on the car thing. 

The prebuilt model + runtime bundle lives on **HuggingFace**
([`mira-thing/mira-voice`](https://huggingface.co/mira-thing/mira-voice))

## Layout
- `ARCHITECTURE.md` - full breakdown of the pipeline
- `BUILD.md` - how to build each binary (for the car thing specifically)
- `THIRD_PARTY_LICENSES` - attribution for the upstream artifacts in the huggingface bundle.

## Quickstart
```sh
bash fetch-artifacts.sh 
# or rebuild from source
bash collect-artifacts.sh 
```
