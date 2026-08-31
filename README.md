# Mira Voice Stack

On-device smart voice controls for the Spotify Car Thing: wake ("hey mira") -> ASR ->
cascade re-rank against your library -> play/control.
100% local on the car thing.

Part of [Mira](https://github.com/mira-thing).

The prebuilt model + runtime bundle lives on **HuggingFace**
([`mira-thing/mira-voice`](https://huggingface.co/mira-thing/mira-voice))

## Support

Mira is free and open source. If you'd like to support development, you can do so on [GitHub Sponsors](https://github.com/sponsors/MustakimK) or [Ko-fi](https://ko-fi.com/MustakimK). Sponsors get early access to betas and access to the dev chat, both set up through [Discord](https://discord.gg/SR2Pne7EPM). Every bit genuinely helps and it's what makes this sustainable to keep working on.

## Related projects

- [`mira-ui`](https://github.com/mira-thing/mira-ui) - Vite + React UI
- [`mira-daemon`](https://github.com/mira-thing/mira-daemon) - daemon
- [`mira-firmware`](https://github.com/mira-thing/mira-firmware) - image builder
- [`mira-releases`](https://github.com/mira-thing/mira-releases) - prebuilt firmware images
- [`mira-voice`](.) - on-device voice stack (this repo)

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
