#!/usr/bin/env python3
# generate the synthetic eval set: accent and condition over catalog names, via Parler-TTS + device-mic
# run with the Parler venv
import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

INDEX = HERE / "index" / "phonetic_index_device.json"
N_TRACKS, N_ARTISTS, N_PLAYLISTS, N_ALBUMS = 90, 24, 18, 18


def _norm(s):
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _speakable(name):
    # a clean sayable name: 1-6 words, has letters, studio suffixes dropped
    base = re.split(r"\s*-\s*(?:demo|remaster|remix|live|edit|version|instrumental)", name, flags=re.I)[0]
    w = base.split()
    return base if (1 <= len(w) <= 6 and re.search(r"[A-Za-zÀ-￿]", base)) else None


def _transcribable(s):
    # a plausibly-transcribable title: ASCII-ish, 1-5 words, mostly alphabetic
    if not s or not re.match(r"^[A-Za-z0-9][A-Za-z0-9 '&,.?-]*$", s):
        return False
    w = s.split()
    if not (1 <= len(w) <= 5):
        return False
    body = s.replace(" ", "")
    return sum(c.isalpha() for c in body) >= 0.7 * len(body)


def _pick_names():
    # sample sayable names from the device catalog index to synthesise commands
    idx = json.load(open(INDEX))["index"]
    rng = random.Random(20260621)
    seen = {}
    tracks = []
    pool = idx["tracks"][:]
    rng.shuffle(pool)
    for e in pool:
        st, sa = _speakable(e.get("name", "")), _speakable(e.get("artist", ""))
        a = _norm(e.get("artist", ""))
        if not st or not sa or not _transcribable(st) or seen.get(a, 0) >= 2:
            continue
        seen[a] = seen.get(a, 0) + 1
        tracks.append([st, sa, e["uri"]])
        if len(tracks) >= N_TRACKS:
            break

    def simple(key, n):
        items = [e for e in idx[key] if _speakable(e["name"]) and _transcribable(e["name"])]
        rng.shuffle(items)
        return items[:n]

    artists = [[_speakable(e["name"]), e["uri"]] for e in simple("artists", N_ARTISTS)]
    playlists = [[_speakable(e["name"]), e["uri"]] for e in simple("playlists", N_PLAYLISTS)]
    albums = [[_speakable(e["name"]), e.get("artist", ""), e["uri"]] for e in simple("albums", N_ALBUMS)]
    return tracks, artists, playlists, albums


TRACKS, ARTISTS, PLAYLISTS, ALBUMS = _pick_names()

ACCENTS = ["us", "gb", "in", "au", "latam"]
CONDS = ["clean", "muffled", "quiet", "musicfan", "louderfan"]

# nam -scorable clips per (accent, cond)skewed to the noise gap
MATRIX = {
    "us":    {"clean": 16, "muffled": 20, "quiet": 20, "musicfan": 28, "louderfan": 36},
    "gb":    {"clean": 12, "muffled": 16, "quiet": 16, "musicfan": 24, "louderfan": 32},
    "in":    {"clean": 12, "muffled": 16, "quiet": 16, "musicfan": 24, "louderfan": 32},
    "au":    {"clean": 10, "muffled": 14, "quiet": 14, "musicfan": 20, "louderfan": 28},
    "latam": {"clean": 12, "muffled": 16, "quiet": 16, "musicfan": 24, "louderfan": 32},
}
N_CONTROL = 8
N_STRESS = 10

# intent mix within a name scorable cell
INTENT_MIX = (["track"] * 45 + ["artist"] * 15 + ["playlist"] * 8 + ["album"] * 8 + ["bare"] * 24)

# Parler accent descriptions
ACCENT_DESC = {
    "us":    "An American {g} speaks {e}, close to the microphone, clean recording.",
    "gb":    "A British {g} speaks {e}, close to the microphone, clean recording.",
    "in":    "An Indian {g} speaks {e} in clear Indian-accented English, close to the microphone.",
    "au":    "An Australian {g} speaks {e}, close to the microphone, clean recording.",
    "latam": "A {g} speaks English with a Latin American Spanish accent, {e}, close to the microphone.",
}
GENDERS = ["man", "woman"]
EXPR = ["naturally", "fairly fast", "slightly expressively", "calmly", "casually"]
TEMPS = [0.7, 1.0, 1.3]

CONTROLS = [("pause", "pause"), ("stop", "pause"), ("skip this song", "next"),
            ("next track", "next"), ("next song", "next"), ("go back", "prev"),
            ("previous track", "prev"), ("resume", "resume"), ("shuffle", "shuffle"),
            ("turn it up", "noop")]


def track_phrase(rng, t, a, u):
    tmpl = rng.choice(["play {t} by {a}", "play {t} by {a}", "put on {t} by {a}",
                       "play the song {t} by {a}"])
    return tmpl.format(t=t, a=a), "track", t, a, u


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 1.0
    plan_only = "--plan-only" in sys.argv
    rng = random.Random(20260621)

    rows = []
    
    cur = {"track": 0, "artist": 0, "playlist": 0, "album": 0}
    pools = {"track": TRACKS, "artist": ARTISTS, "playlist": PLAYLISTS, "album": ALBUMS}

    def nxt(kind):
        p = pools[kind]
        item = p[cur[kind] % len(p)]
        cur[kind] += 1
        return item

    def build_phrase(kind):
        if kind == "track":
            t, a, u = nxt("track"); return track_phrase(rng, t, a, u)
        if kind == "artist":
            a, u = nxt("artist")
            tmpl = rng.choice(["play {a}", "play {a}", "play {a} radio", "play {a} discography"])
            return tmpl.format(a=a), "artist", a, "", u
        if kind == "playlist":
            n, u = nxt("playlist")
            tmpl = rng.choice(["play my {n} playlist", "play the {n} playlist"])
            return tmpl.format(n=n), "playlist", n, "", u
        if kind == "album":
            t, a, u = nxt("album")
            return f"play the album {t}", "album", t, a, u
        # bare track
        t, a, u = nxt("track")
        return f"play {t}", "bare", t, a, u

    def stress_phrase(sub):
        if sub == "dropverb":
            t, a, u = nxt("track"); return f"{t} by {a}", "track", t, a, u  # no "play"
        if sub == "baregarbled":
            t, a, u = nxt("track"); return f"{t} {a}", "bare", t, a, u       # no "by"
        if sub == "short":
            if rng.random() < 0.5:
                t, a, u = nxt("track"); return t.split()[0], "bare", t, a, u  # 1-word
            a, u = nxt("artist"); return a.split()[0], "bare", a, "", u
        # filler heavy
        t, a, u = nxt("track")
        lead = rng.choice(["uh can you put on", "i wanna hear", "could you play", "um play"])
        return f"{lead} {t} by {a}", "track", t, a, u

    def add(accent, cond, intent, stress, spoken, e1, e2, euri, name_set):
        g = rng.choice(GENDERS); e = rng.choice(EXPR); temp = rng.choice(TEMPS)
        rows.append({"accent": accent, "cond": cond, "intent": intent, "stress": stress,
                     "spoken": spoken, "expected_1": e1, "expected_2": e2, "expected_uri": euri,
                     "name_set": name_set, "tts_voice": f"{accent}_{g}", "tts_temp": temp,
                     "_desc": ACCENT_DESC[accent].format(g=g, e=e)})

    # 1. name scorable matrix
    for accent in ACCENTS:
        for cond in CONDS:
            n = max(1, round(MATRIX[accent][cond] * scale))
            for _ in range(n):
                spoken, intent, e1, e2, u = build_phrase(rng.choice(INTENT_MIX))
                add(accent, cond, intent, "", spoken, e1, e2, u, "library")

    # 2. control block
    cond_w = ["clean", "muffled", "quiet", "musicfan", "musicfan", "louderfan", "louderfan"]
    for accent in ACCENTS:
        nc = max(1, round(N_CONTROL * scale))
        for i in range(nc):
            spoken, action = CONTROLS[i % len(CONTROLS)]
            add(accent, rng.choice(cond_w), "control", "control", spoken, action, "", "", "control")

    # 3. verb stress block
    subs = ["dropverb", "dropverb", "baregarbled", "baregarbled", "short", "filler"]
    for accent in ACCENTS:
        ns = max(1, round(N_STRESS * scale))
        for i in range(ns):
            sub = subs[i % len(subs)]
            spoken, intent, e1, e2, u = stress_phrase(sub)
            add(accent, rng.choice(cond_w), intent, sub, spoken, e1, e2, u, "library")

    counter = {}
    for r in rows:
        k = (r["accent"], r["cond"]); counter[k] = counter.get(k, 0) + 1
        r["idx"] = counter[k]
        r["clip"] = f"clips/{r['accent']}/{r['cond']}/{r['idx']:03d}.wav"
        r["_raw"] = f"raw_tts/{r['accent']}/{r['cond']}/{r['idx']:03d}.wav"

    cols = ["accent", "cond", "idx", "clip", "intent", "stress", "spoken",
            "expected_1", "expected_2", "expected_uri", "name_set", "tts_voice", "tts_temp"]
    with open(HERE / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"PLAN: {len(rows)} clips (scale {scale}) -> manifest.csv")
    from collections import Counter
    print("  by cond:", dict(Counter(r["cond"] for r in rows)))
    print("  by accent:", dict(Counter(r["accent"] for r in rows)))
    print("  by intent:", dict(Counter(r["intent"] for r in rows)))
    print("  by stress:", dict(Counter(r["stress"] for r in rows if r["stress"])))
    if plan_only:
        return

    # audio generation: Parler TTS -> device_mic_sim
    import numpy as np, soundfile as sf, torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    from device_mic_sim import apply_device_mic, load_clean, SR

    dev = "cuda:0"; ckpt = "parler-tts/parler-tts-mini-v1"
    print("loading Parler...", flush=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(ckpt).to(dev)
    tok = AutoTokenizer.from_pretrained(ckpt)
    psr = model.config.sampling_rate

    done = 0
    for i, r in enumerate(rows):
        raw_p = HERE / r["_raw"]; clip_p = HERE / r["clip"]
        clip_p.parent.mkdir(parents=True, exist_ok=True); raw_p.parent.mkdir(parents=True, exist_ok=True)
        if clip_p.exists():
            done += 1; continue
        if not raw_p.exists():
            ids = tok(r["_desc"], return_tensors="pt").input_ids.to(dev)
            pids = tok(r["spoken"], return_tensors="pt").input_ids.to(dev)
            torch.manual_seed(1000 + i)
            with torch.no_grad():
                audio = model.generate(input_ids=ids, prompt_input_ids=pids,
                                       do_sample=True, temperature=float(r["tts_temp"]))
            sf.write(raw_p, audio.cpu().numpy().squeeze().astype("float32"), psr)
        y = apply_device_mic(load_clean(str(raw_p)), profile=r["cond"], seed=1000 + i)
        sf.write(clip_p, y, SR, subtype="PCM_16")
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(rows)}", flush=True)
    print(f"AUDIO DONE: {done}/{len(rows)} clips under {HERE}/clips/")


if __name__ == "__main__":
    main()
