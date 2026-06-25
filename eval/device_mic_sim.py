#!/usr/bin/env python3
# device-mic far-field PDM simulator: turn a clean TTS clip into something like the Car Thing's
# far-field PDM capture (reverb + music/fan bed + PDM HF tilt + low-gain quant) per condition profile.
# optional assets (degrade gracefully if absent): RIR wavs in <assets>/rirs, music wavs in <assets>/music.
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

HERE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("EVAL_ASSETS", HERE / "assets"))
RIR_DIR = ASSETS / "rirs"
BG_DIR = ASSETS / "music"
SR = 16000

# loaded-once asset caches
_RIRS = {}
_BG = []


def _load_rirs():
    if _RIRS:
        return
    for p in sorted(RIR_DIR.glob("*.wav")):
        d, sr = sf.read(p)
        if d.ndim > 1:
            d = d.mean(axis=1)
        d = d.astype(np.float32)
        if sr != SR:
            d = signal.resample_poly(d, SR, sr).astype(np.float32)
        # energy-normalize, skip silent
        peak = np.abs(d).max()
        if peak < 1e-6:
            continue
        d = d / peak
        _RIRS[p.name] = d


def _load_bg(limit=300):
    if _BG:
        return
    for p in sorted(BG_DIR.glob("*.wav"))[:limit]:
        d, sr = sf.read(p)
        if d.ndim > 1:
            d = d.mean(axis=1)
        d = d.astype(np.float32)
        if sr != SR:
            d = signal.resample_poly(d, SR, sr).astype(np.float32)
        if len(d) < SR:
            continue
        _BG.append(d)


# per-clip condition profiles: rir_names biases the room, snr_db is speech-to-bed SNR (lower = noisier),
# tilt_db is the PDM HF de-emphasis; lpf_hz/occl_db/pre_gain_db/gain_div are optional per-profile keys.
PROFILES = {
    "clean": dict(
        rir_names=["NONE", "NONE", "h001", "h004", "h010", "h012", "h011_Car"],
        snr_db=(20.0, 34.0), fan_snr_db=(22.0, 38.0), music_p=0.12,
        tilt_db=(6.0, 13.0),
    ),
    "musicfan": dict(
        rir_names=["h011_Car", "h011_Car", "h004", "h010", "h017", "h012"],
        snr_db=(10.0, 18.0), fan_snr_db=(12.0, 20.0), music_p=0.8,
        tilt_db=(9.0, 16.0),
    ),
    "louderfan": dict(  # louder fan + lyrical song bed (the noise-gap condition)
        rir_names=["h011_Car", "h011_Car", "h011_Car", "h004", "h017"],
        snr_db=(6.0, 13.0), fan_snr_db=(8.0, 15.0), music_p=0.92,
        tilt_db=(11.0, 19.0),
    ),
    # muffled: occlusion/off-axis transfer path, intelligible speech but gutted HF + bandlimit + boxy resonance
    "muffled": dict(
        rir_names=["h011_Car", "h004", "h010", "h012", "h017"],
        snr_db=(12.0, 22.0), fan_snr_db=(14.0, 24.0), music_p=0.45,
        tilt_db=(16.0, 24.0),
        lpf_hz=(2200.0, 3400.0),
        occl_db=(3.0, 7.0),
    ),
    # quiet: soft/far speech attenuated before noise+quant so the low-gain quant floor bites harder
    "quiet": dict(
        rir_names=["h011_Car", "h011_Car", "h004", "h010", "h017"],
        snr_db=(8.0, 16.0), fan_snr_db=(10.0, 18.0), music_p=0.55,
        tilt_db=(10.0, 18.0),
        pre_gain_db=(-14.0, -7.0),
        gain_div=48.0,
    ),
}


def _pick_rir(names):
    n = random.choice(names)
    if n == "NONE":
        return None
    for k in _RIRS:
        if k.startswith(n):
            return _RIRS[k]
    return None


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _fan_noise(n):
    # broadband fan/HVAC noise: pink-ish (1/f) with a low-mid hump
    white = np.random.randn(n).astype(np.float32)
    b, a = signal.butter(1, 0.18, btype="low")
    pink = signal.lfilter(b, a, white).astype(np.float32)
    fan = 0.65 * pink + 0.35 * white
    return fan / (_rms(fan))


def _pdm_tilt(x, tilt_db):
    # HF de-emphasis shelf above ~3 kHz + sub-100 Hz rumble cut (the measured PDM transfer)
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g_db = np.zeros_like(f)
    hi = f >= 3000.0
    g_db[hi] = -tilt_db * np.clip((np.log2(f[hi] / 3000.0)) / np.log2(8000.0 / 3000.0), 0, 1)
    lo = f < 90.0
    g_db[lo] = -18.0
    g = 10.0 ** (g_db / 20.0)
    return np.fft.irfft(X * g, n=n).astype(np.float32)


def _muffle(x, lpf_hz, occl_db):
    # occlusion: 4th-order Butterworth LPF then a peaking-EQ bump ~550 Hz (cupped-mic color)
    nyq = SR / 2.0
    fc = min(max(float(lpf_hz), 500.0), nyq * 0.98)
    b, a = signal.butter(4, fc / nyq, btype="low")
    y = signal.filtfilt(b, a, x).astype(np.float32)
    f0, Q = 550.0, 1.0
    A = 10.0 ** (float(occl_db) / 40.0)
    w0 = 2.0 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2.0 * Q)
    cw = np.cos(w0)
    b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A
    a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A
    y = signal.lfilter([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0], y).astype(np.float32)
    return y


def _low_gain_quant(x, gain_div=24.0):
    # scale down (low analog gain), S16-quantize, scale back up: bakes in the low-gain quant floor
    lo = x / gain_div
    lo = np.clip(lo, -1.0, 1.0)
    q = np.round(lo * 32767.0).astype(np.int16).astype(np.float32) / 32767.0
    return q * gain_div


def apply_device_mic(speech, profile="musicfan", seed=None):
    # speech: float32 @16k ~unit-peak; returns float32 @16k peak-normalized to 0.9
    if seed is not None:
        random.seed(seed); np.random.seed(seed % (2**32))
    _load_rirs(); _load_bg()
    prof = PROFILES[profile]
    x = speech.astype(np.float32).copy()
    if np.abs(x).max() > 1e-6:
        x = x / np.abs(x).max()

    # 1. RIR convolution (far-field reverb)
    rir = _pick_rir(prof["rir_names"])
    if rir is not None:
        x = signal.fftconvolve(x, rir)[: len(speech) + len(rir)].astype(np.float32)
        x = x[: len(speech) + 800]
    pre_gain_db = prof.get("pre_gain_db")
    if pre_gain_db is not None:
        x = x * (10.0 ** (random.uniform(*pre_gain_db) / 20.0))
    sp_rms = _rms(x)

    # 2a. music bed under the speech
    if random.random() < prof["music_p"] and _BG:
        bed = random.choice(_BG)
        if len(bed) < len(x):
            bed = np.tile(bed, int(np.ceil(len(x) / len(bed))))
        off = random.randint(0, max(0, len(bed) - len(x)))
        bed = bed[off: off + len(x)].astype(np.float32)
        snr = random.uniform(*prof["snr_db"])
        target = sp_rms / (10.0 ** (snr / 20.0))
        bed = bed * (target / _rms(bed))
        x = x[: len(bed)] + bed

    # 2b. fan/broadband noise under the speech
    fan = _fan_noise(len(x))
    fan = _pdm_tilt(fan, random.uniform(2.0, 6.0))
    fsnr = random.uniform(*prof["fan_snr_db"])
    ftarget = sp_rms / (10.0 ** (fsnr / 20.0))
    fan = fan * (ftarget / _rms(fan))
    x = x + fan

    # 3. PDM mic transfer (HF tilt + rumble cut)
    x = _pdm_tilt(x, random.uniform(*prof["tilt_db"]))

    # 3b. muffled: occlusion bandlimit + boxy resonance
    lpf_hz = prof.get("lpf_hz")
    occl_db = prof.get("occl_db")
    if lpf_hz is not None and occl_db is not None:
        x = _muffle(x, random.uniform(*lpf_hz), random.uniform(*occl_db))

    pk = np.abs(x).max()
    if pk > 1.0:
        x = x / pk

    # 4. low-gain -> S16 quant -> renormalize
    x = _low_gain_quant(x, gain_div=prof.get("gain_div", 24.0))

    # 5+6. S16 @16k, peak-normalize to 0.9
    pk = np.abs(x).max()
    if pk < 1e-6:
        pk = 1.0
    x = x * (0.9 / pk)
    x = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float32) / 32767.0
    return x


def load_clean(path):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d.astype(np.float32)
    if sr != SR:
        d = signal.resample_poly(d, SR, sr).astype(np.float32)
    return d


if __name__ == "__main__":
    import sys
    inp, outp, prof = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "musicfan")
    y = apply_device_mic(load_clean(inp), profile=prof, seed=1234)
    sf.write(outp, y, SR, subtype="PCM_16")
    print(f"wrote {outp}  prof={prof}  rms={_rms(y):.5f} peak={np.abs(y).max():.4f}")
