# Mira Voice Stack

On-device smart voice controls for the Spotify Car Thing: wake ("hey mira") -> ASR ->
cascade re-rank against your library -> play/control.
100% local on the car thing.

Part of [Mira](https://github.com/mira-thing).

The prebuilt model + runtime bundle lives on **HuggingFace**
([`mira-thing/mira-voice`](https://huggingface.co/mira-thing/mira-voice))

## Related projects

- [`mira-ui`](https://github.com/mira-thing/mira-ui) - Vite + React UI
- [`mira-daemon`](https://github.com/mira-thing/mira-daemon) - daemon
- [`mira-firmware`](https://github.com/mira-thing/mira-firmware) - image builder
- [`mira-releases`](https://github.com/mira-thing/mira-releases) - prebuilt firmware images
- [`mira-voice`](.) - on-device voice stack (this repo)

## Support

Mira is free and open source. If you'd like to support development, you can do so on [GitHub Sponsors](https://github.com/sponsors/MustakimK) or [Ko-fi](https://ko-fi.com/MustakimK). Questions and updates are on [Discord](https://discord.gg/SR2Pne7EPM).

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

## License

Apache 2.0, see [LICENSE](LICENSE).

> "Spotify" and "Car Thing" are trademarks of Spotify AB. This software is not affiliated with or endorsed by Spotify AB.
