#!/usr/bin/env python3
# run a sherpa-onnx offline Zipformer on all eval clips
import glob
import os
import sys

import sherpa_onnx
import soundfile as sf

EVAL = os.path.dirname(os.path.abspath(__file__)) 
model_dir, tag = sys.argv[1], sys.argv[2]
hotwords = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
score = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
method = sys.argv[5] if len(sys.argv) > 5 else "modified_beam_search"

enc = glob.glob(f"{model_dir}/encoder*int8.onnx")[0]
dec = glob.glob(f"{model_dir}/decoder*int8.onnx")[0]
joi = glob.glob(f"{model_dir}/joiner*int8.onnx")[0]
kw = dict(encoder=enc, decoder=dec, joiner=joi, tokens=f"{model_dir}/tokens.txt",
          num_threads=4, decoding_method=method)
if hotwords:
    kw.update(modeling_unit="bpe", bpe_vocab=f"{model_dir}/bpe.vocab",
              hotwords_file=hotwords, hotwords_score=score)
rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)

clips = sorted(glob.glob(f"{EVAL}/clips/**/*.wav", recursive=True))
n = 0
with open(f"{EVAL}/transcripts_{tag}.tsv", "w") as out:
    for c in clips:
        samples, sr = sf.read(c, dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]
        s = rec.create_stream()
        s.accept_waveform(sr, samples)
        rec.decode_stream(s)
        rel = os.path.relpath(c, EVAL)
        out.write(f"{rel}\t{s.result.text.strip()}\n")
        n += 1
print(f"{tag}: wrote {n} -> transcripts_{tag}.tsv (hotwords={hotwords})")
