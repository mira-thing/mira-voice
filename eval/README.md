# eval - synthetic accuracy eval

A multi condition synthetic eval that scores the cascade resolver against the real device catalog
index. It avoids two optimism traps: a tiny made-up index, and real phonetic confusers being
absent.

## Why it's fair
- Realistic index
- Catalog names
- Real accents + noise

## How to run
```sh
# 1. generate audio: sample catalog names -> accent TTS -> device-mic sim -> clips/ + manifest.csv
#    SCALE in (0,1]; needs the Parler venv (device_mic_sim.py is in-repo; assets optional, see below)
python gen_eval.py 1.0

# 2. transcribe host-side with the sherpa Zipformer
python host_transcribe_zipformer.py <sherpa_model_dir> <tag> 

# 3. score against the real catalog index
python score_eval.py transcripts_<tag>.tsv
```
`sweep_threshold.py transcripts_<tag>.tsv`

## Assets (optional)
- `device_mic_sim.py` mixes in reverb + a music bed if you supply wavs (else it runs without them)
- `assets/rirs/` - room impulse responses (e.g. MIT RIR survey)
- `assets/music/` - background music (e.g. MUSAN music)

Set `EVAL_ASSETS` to override the location. `assets/` is gitignored.

## Metrics (`score_eval.py`)
- local-accept - resolved to the right URI in-library
- right-item-found (any tier) - the rerank's top pick was correct even if below threshold
- raw ASR name-survival - did the expected name words survive the ASR
