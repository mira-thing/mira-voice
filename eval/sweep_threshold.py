#!/usr/bin/env python3
# sweep the accept threshold over the eval set: recall vs false-accept vs abstain at each T
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("se", str(HERE / "score_eval.py"))
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)


def main():
    import json
    tsvs = sys.argv[1:]
    if not tsvs:
        print("usage: sweep_threshold.py <transcripts.tsv> [...]"); return
    idx = se.Index(json.load(open(se.INDEX_JSON))["index"])
    hyps = defaultdict(list)
    for p in tsvs:
        for ln in open(p):
            if "\t" in ln:
                c, t = ln.rstrip("\n").split("\t", 1)
                hyps[c].append(t.strip())
    rows = list(csv.DictReader(open(se.MANIFEST)))

    # per clip: (best_score, correct_pick, has_uri)  for name scorable clips only
    recs = []
    by_cond = defaultdict(list)
    by_intent = defaultdict(list)
    ctrl_ok = ctrl_n = 0
    for r in rows:
        clip = r["clip"]; hh = hyps.get(clip) or []
        if not hh:
            continue
        intent = r["intent"]; euri = r.get("expected_uri", ""); e1 = r.get("expected_1", "")
        d = idx.resolve(hh)
        if intent == "control":
            ctrl_n += 1
            ctrl_ok += int(d["tier"] == "control" and d.get("action") == e1)
            continue
        sc = d.get("score", None)
        picked_uri = d.get("uri", "")
        if euri:
            correct = (picked_uri == euri)
        else:
            correct = (d.get("name", "").lower() == e1.lower())
        has_pick = bool(picked_uri) or bool(d.get("name"))
        rec = (sc, correct, has_pick, r["cond"], intent)
        recs.append(rec)
        if sc is not None:
            by_cond[r["cond"]].append(rec)
            by_intent[intent].append(rec)

    N = len(recs)
    print(f"name-scorable clips: {N} | controls: {ctrl_ok}/{ctrl_n} routed = {100*ctrl_ok/max(ctrl_n,1):.0f}%")
    print(f"tsvs: {len(tsvs)} ({', '.join(Path(t).name for t in tsvs)})\n")

    def sweep(label, data):
        n = len(data)
        print(f"== threshold sweep: {label} (n={n}) ==")
        print(f"  {'T':>5}  {'recall':>14}  {'false-accept':>14}  {'abstain':>12}")
        for T in [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.70]:
            racc = sum(1 for sc, ok, hp, *_ in data if sc is not None and sc <= T and ok)
            facc = sum(1 for sc, ok, hp, *_ in data if sc is not None and sc <= T and hp and not ok)
            abst = n - sum(1 for sc, ok, hp, *_ in data if sc is not None and sc <= T and hp)
            mark = "  <- prod 0.42" if abs(T - se.ACCEPT) < 1e-9 else ""
            print(f"  {T:>5.2f}  {racc:>4}/{n} = {100*racc/max(n,1):3.0f}%  "
                  f"{facc:>4}/{n} = {100*facc/max(n,1):3.0f}%  {abst:>4}/{n} = {100*abst/max(n,1):3.0f}%{mark}")
        print()

    sweep("ALL name-scorable", recs)
    for c in ["clean", "quiet", "muffled", "musicfan", "louderfan"]:
        if by_cond.get(c):
            sweep(f"cond={c}", by_cond[c])
    for it in ["track", "bare", "artist", "album", "playlist"]:
        if by_intent.get(it):
            sweep(f"intent={it}", by_intent[it])


if __name__ == "__main__":
    main()
