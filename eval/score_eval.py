#!/usr/bin/env python3
# score the eval set against the cascade resolver using the real device catalog index
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from piper_phonemize import phonemize_espeak

HERE = Path(__file__).resolve().parent
INDEX_JSON = HERE / "index" / "phonetic_index_device.json"
MANIFEST = HERE / "manifest.csv"

ACCEPT = 0.42
TRACK_ANCHORED_FLOOR = 0.60 
BARE_FLOOR = 0.30 
VERB_THRESHOLD = 0.34 
W_TRACK, W_ARTIST = 1.0, 0.5
BARE_MAX_PREFIX_WORDS = 6
BARE_PREFIX_PENALTY = 0.01

# control phrases checked longest first
CONTROL_PHRASES = [
    ("turn it up", "volup"), ("turn it down", "voldown"),
    ("volume up", "volup"), ("volume down", "voldown"),
    ("turn up", "volup"), ("turn down", "voldown"),
    ("go back", "prev"),
    ("like this", "like"), ("save this", "like"), ("love this", "like"),
    ("unlike this", "unlike"), ("unsave this", "unlike"), ("dislike this", "unlike"),
    ("pause", "pause"), ("stop", "pause"), ("resume", "resume"),
    ("skip", "next"), ("next", "next"), ("previous", "prev"), ("back", "prev"),
    ("shuffle", "shuffle"),
]
CONTROL_PARTICLES = {"this", "that", "it", "song", "track", "please", "now", "one",
                     "the", "ahead", "again"}
LEAD_PLAY_VARIANTS = {"clay", "pray", "slay", "played", "plays", "prey", "flay",
                      "blay", "plei", "pleh"}
LIKED_FILLER = {"play", "my", "the", "a", "some", "of", "to", "songs", "song", "tracks",
                "track", "music", "liked", "licked", "like", "list", "all", "collection",
                "saved", "favourites", "favorites"}
RANDOM_FILLER = {"play", "a", "an", "some", "the", "my", "song", "songs", "track", "tracks",
                 "music", "something", "any", "random", "anything", "shuffle"}
FILLER = {"play", "please", "the", "a", "an", "of", "for", "his", "her", "to", "put",
          "on", "some", "my", "music", "song", "track", "by"}

STRESS = "ˈˌ"
_memo = {}
def ipa(t):
    t = (t or "").lower().strip()
    if t in _memo:
        return _memo[t]
    s = "".join(p for sent in phonemize_espeak(t, "en-us") for p in sent)
    s = "".join(c for c in s if c not in STRESS and not c.isspace())
    _memo[t] = s
    return s

def lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la or not lb:
        return la or lb
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]; ca = a[i - 1]
        for j in range(1, lb + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1])))
        prev = cur
    return prev[-1]

def nd(a, b):
    return lev(a, b) / max(len(a), len(b), 1)

_nonword = re.compile(r"[^\w\s]")
_ws = re.compile(r"\s+")
def normalize(h):
    return _ws.sub(" ", _nonword.sub(" ", (h or "").lower())).strip()

def strip_filler(s):
    return " ".join(w for w in (s or "").split() if w not in FILLER).strip()

def neg_prefix(w):
    return w.startswith("un") or w.startswith("dis")


def word_match(a, b):
    if a == b:
        return True
    if neg_prefix(a) != neg_prefix(b):
        return False
    ia, ib = ipa(a), ipa(b)
    return bool(ia) and bool(ib) and nd(ia, ib) <= VERB_THRESHOLD


def match_control(words):
    if not words:
        return None
    for phrase, action in CONTROL_PHRASES:
        pw = phrase.split()
        n = len(pw)
        if len(words) < n:
            continue
        if all(word_match(words[i], pw[i]) for i in range(n)) and all(w in CONTROL_PARTICLES for w in words[n:]):
            return action
    return None


def matches_liked(words):
    if not any(w in ("liked", "licked") for w in words):
        return False
    return all(w in LIKED_FILLER for w in words)


def matches_random(t):
    if t in ("surprise me", "surprise"):
        return True
    if "random" not in t:
        return False
    return all(w in RANDOM_FILLER for w in t.split())


def queue_name(t, words):
    if words[0] == "add" and t.endswith(" queue"):
        inner = t[: -len(" queue")]
        if inner.startswith("add "):
            inner = inner[4:]
        for suf in (" to the", " to my", " to"):
            if inner.endswith(suf):
                inner = inner[: -len(suf)]
        inner = inner.strip()
        if inner:
            return inner
    if len(words) > 1:
        w = words[0]
        if w in ("queue", "cue"):
            return " ".join(words[1:])
        a, b = ipa(w), ipa("queue")
        if a and b and nd(a, b) <= VERB_THRESHOLD:
            return " ".join(words[1:])
    return None


def classify(h):
    t = normalize(h)
    if not t:
        return ("bare", {})
    words = t.split()
    a = match_control(words)
    if a:
        return ("control", {"action": a})
    if len(words) > 1 and words[0] in LEAD_PLAY_VARIANTS:
        words[0] = "play"
        t = " ".join(words)
    if matches_liked(words):
        return ("playliked", {})
    if matches_random(t):
        return ("random", {})
    qn = queue_name(t, words)
    if qn is not None:
        gk, gs = classify_grammar(qn)
        if gk == "track":
            return ("queue", {"track": gs.get("track"), "artist": gs.get("artist")})
        return ("queue", {"name": gs.get("name")})
    return classify_grammar(t)


def classify_grammar(t):
    t = t.strip()
    if not t:
        return ("bare", {})
    if " by " in t:
        left, right = t.split(" by ", 1)
        return ("track", {"track": strip_filler(left), "artist": strip_filler(right)})
    if (t.endswith(" radio") or t.endswith(" discography")
            or " everything by " in t or t.endswith(" songs")):
        name = t
        for suf in (" radio", " discography", " songs"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        name = name.replace("everything by", "")
        return ("artist", {"name": strip_filler(name)})
    if "playlist" in t or t.startswith("play my "):
        return ("playlist", {"name": strip_filler(t.replace("playlist", ""))})
    if "album" in t:
        return ("album", {"name": strip_filler(t.replace("album", ""))})
    return ("bare", {"name": strip_filler(t)})


class Index:
    def __init__(self, idx):
        self.tracks = [(e["name"], e.get("artist", ""), e["uri"], ipa(e["name"]), ipa(e.get("artist", "")))
                       for e in idx["tracks"]]
        self.artists = [(e["name"], e["uri"], ipa(e["name"])) for e in idx["artists"]]
        self.playlists = [(e["name"], e["uri"], ipa(e["name"])) for e in idx["playlists"]]
        self.albums = [(e["name"], e.get("artist", ""), e["uri"], ipa(e["name"])) for e in idx["albums"]]

    def match_track(self, track, artist):
        if not track:
            return None
        qt = ipa(track)
        if not qt:
            return None
        qa = ipa(artist) if artist else None
        best = None
        for name, art, uri, ti, ai in self.tracks:
            ad = nd(qa, ai) if qa else 0.0
            sc = W_TRACK * nd(qt, ti) + W_ARTIST * ad
            if best is None or sc < best[0]:
                best = (sc, "track", name, art, uri, bool(qa), False)
        return best

    def _simple(self, name, idx, kind):
        if not name:
            return None
        qi = ipa(name)
        if not qi:
            return None
        best = None
        for row in idx:
            sc = nd(qi, row[-1])
            if best is None or sc < best[0]:
                if kind == "album":
                    best = (sc, kind, row[0], row[1], row[2], False, False)
                else:
                    best = (sc, kind, row[0], "", row[1], False, False)
        return best

    def match_artist(self, name):
        return self._simple(name, self.artists, "artist")

    def match_playlist(self, name):
        return self._simple(name, self.playlists, "playlist")

    def match_album(self, name):
        return self._simple(name, self.albums, "album")

    def match_bare(self, name):
        if not name:
            return None
        best = None
        def consider(m):
            nonlocal best
            if m and (best is None or m[0] < best[0]):
                best = m
        words = name.split()[:BARE_MAX_PREFIX_WORDS]
        for n in range(len(words), 0, -1):
            qi = ipa(" ".join(words[:n]))
            if not qi:
                continue
            pen = (len(words) - n) * BARE_PREFIX_PENALTY
            for nm, art, uri, ti, ai in self.tracks:
                sc = nd(qi, ti) + pen
                if best is None or sc < best[0]:
                    best = (sc, "track", nm, art, uri, False, False)
        consider(self.match_artist(name))
        consider(self.match_playlist(name))
        consider(self.match_album(name))
        if best is not None: 
            best = best[:6] + (True,)
        return best

    @staticmethod
    def _lower(a, b): 
        if a is None:
            return b
        if b is None:
            return a
        return b if b[0] < a[0] else a

    def match_for_intent(self, kind, slots):
        if kind == "track":
            return self.match_track(slots.get("track"), slots.get("artist"))
        if kind == "queue":
            if slots.get("track"):
                return self.match_track(slots.get("track"), slots.get("artist"))
            return self.match_bare(slots.get("name"))
        if kind == "artist":
            return self._lower(self.match_artist(slots.get("name")), self.match_bare(slots.get("name")))
        if kind == "playlist":
            return self._lower(self.match_playlist(slots.get("name")), self.match_bare(slots.get("name")))
        if kind == "album":
            return self._lower(self.match_album(slots.get("name")), self.match_bare(slots.get("name")))
        return self.match_bare(slots.get("name"))

    @staticmethod
    def _floor_for(m):
        _, kind, _, _, _, anchored, bare = m
        if kind == "track" and anchored and TRACK_ANCHORED_FLOOR > ACCEPT:
            return TRACK_ANCHORED_FLOOR
        if bare and BARE_FLOOR < ACCEPT:
            return BARE_FLOOR
        return ACCEPT

    def resolve(self, hyps):
        votes = {}
        q_acc = q_any = None
        queue_seen = play_liked = rand = False
        best_acc = best_any = None
        for h in hyps:
            if not (h or "").strip():
                continue
            kind, slots = classify(h)
            if kind == "control":
                votes[slots["action"]] = votes.get(slots["action"], 0) + 1
                continue
            if kind == "playliked":
                play_liked = True
                continue
            if kind == "random":
                rand = True
                continue
            if kind == "queue":
                queue_seen = True
                m = self.match_for_intent(kind, slots)
                if m:
                    if q_any is None or m[0] < q_any[0]:
                        q_any = m
                    if m[0] <= self._floor_for(m) and (q_acc is None or m[0] < q_acc[0]):
                        q_acc = m
                continue
            m = self.match_for_intent(kind, slots)
            if not m:
                continue
            if best_any is None or m[0] < best_any[0]:
                best_any = m
            if m[0] <= self._floor_for(m) and (best_acc is None or m[0] < best_acc[0]):
                best_acc = m

        if queue_seen:
            if q_acc is not None:
                sc, kind, name, art, uri, _, _ = q_acc
                return {"tier": "queue", "kind": kind, "name": name, "artist": art, "uri": uri, "score": sc}
            d = {"tier": "abstain", "kind": "queue"}
            if q_any is not None:
                d["name"], d["score"] = q_any[2], q_any[0]
            return d
        if play_liked:
            return {"tier": "local", "kind": "collection", "uri": "spotify:collection:tracks", "name": "Liked Songs"}
        if rand:
            return {"tier": "abstain", "kind": "random"}
        if best_acc is not None:
            sc, kind, name, art, uri, _, _ = best_acc
            return {"tier": "local", "kind": kind, "name": name, "artist": art, "uri": uri, "score": sc}
        if votes:
            return {"tier": "control", "action": max(votes, key=votes.get)}
        if best_any is None:
            return {"tier": "search"}
        sc, kind, name, art, uri, _, _ = best_any
        return {"tier": "search", "kind": kind, "name": name, "uri": uri, "score": sc}


def survived(tx, e1, e2):
    keys = [w for w in re.findall(r"\w+", (e1 + " " + e2).lower()) if len(w) > 2]
    toks = re.findall(r"\w+", (tx or "").lower())
    return all(any(lev(k, w) <= 1 for w in toks) for k in keys) if keys else True


def main():
    import json
    tsvs = sys.argv[1:] or [str(HERE / "transcripts_tiny.en-q5_1.tsv")]
    idx = Index(json.load(open(INDEX_JSON))["index"])
    print(f"index: {len(idx.tracks)} tracks / {len(idx.artists)} artists / "
          f"{len(idx.playlists)} playlists / {len(idx.albums)} albums | accept {ACCEPT}")

    hyps = defaultdict(list)
    for p in tsvs:
        for ln in open(p):
            if "\t" in ln:
                c, t = ln.rstrip("\n").split("\t", 1)
                hyps[c].append(t.strip())
    rows = list(csv.DictReader(open(MANIFEST)))
    print(f"manifest: {len(rows)} rows | transcripts for {len(hyps)} clips from {len(tsvs)} tsv(s)\n")

    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    def bump(key, ok, surv, scorable, found):
        a = agg[key]
        a[3] += 1; a[2] += int(bool(surv))
        if scorable:
            a[1] += 1; a[0] += int(bool(ok)); a[4] += int(bool(found))

    misses = []
    for r in rows:
        clip = r["clip"]
        hh = hyps.get(clip) or hyps.get(str(Path(clip))) or []
        if not hh:
            continue
        intent = r["intent"]; e1 = r.get("expected_1", ""); e2 = r.get("expected_2", "")
        euri = r.get("expected_uri", "")
        accent = r.get("accent", "?"); cond = r["cond"]; stress = r.get("stress", "") or "-"
        nameset = r.get("name_set", "?")
        d = idx.resolve(hh)
        if intent == "control":
            ok = (d["tier"] == "control" and d.get("action") == e1)
            found = ok; scorable = True; surv = True
        else:
            surv = any(survived(t, e1, e2) for t in hh)
            if euri:
                found = (d.get("uri") == euri) 
                ok = (d["tier"] == "local" and d.get("uri") == euri)
            else:
                found = (d.get("name", "").lower() == e1.lower())
                ok = (d["tier"] == "local" and found)
            scorable = True
        for key in [("ALL",), ("accent", accent), ("cond", cond), ("accent.cond", accent, cond),
                    ("stress", stress), ("nameset", nameset), ("intent", intent)]:
            bump(key, ok, surv, scorable, found)
        if not ok and scorable:
            misses.append((accent, cond, stress, intent, e1, e2, d, hh[0] if hh else ""))

    def show(title, prefix):
        keys = sorted(k for k in agg if k[0] == prefix)
        if not keys:
            return
        print(f"\n== {title} ==")
        print(f"  {'key':22} {'local-accept':>13} {'found(any tier)':>16} {'asr-survive':>12}")
        for k in keys:
            ok, ns, sv, n, fnd = agg[k]
            label = ".".join(str(x) for x in k[1:])
            print(f"  {label:22} {ok}/{ns} = {100*ok/max(ns,1):3.0f}%    {fnd}/{ns} = {100*fnd/max(ns,1):3.0f}%      "
                  f"{sv}/{n} = {100*sv/max(n,1):3.0f}%")

    a = agg[("ALL",)]
    print(f"\n###### OVERALL: local-accept {a[0]}/{a[1]} = {100*a[0]/max(a[1],1):.0f}%  |  "
          f"right-item-found(any tier) {a[4]}/{a[1]} = {100*a[4]/max(a[1],1):.0f}%  |  "
          f"raw ASR name-survival {a[2]}/{a[3]} = {100*a[2]/max(a[3],1):.0f}% ######")
    show("by condition", "cond")
    show("by accent", "accent")
    show("by accent x condition", "accent.cond")
    show("by verb-stress subtype", "stress")
    show("by name_set", "nameset")
    show("by intent", "intent")

    print(f"\n== sample misses ({min(len(misses),25)} of {len(misses)}) ==")
    for accent, cond, stress, intent, e1, e2, d, tx in misses[:25]:
        got = d.get("name", "") + (f" / {d.get('artist','')}" if d.get("artist") else "")
        print(f"  [{accent}/{cond}/{stress}/{intent}] want '{e1}{' / '+e2 if e2 else ''}' "
              f"-> {d['tier']}:{got} (sc {d.get('score','?')})  [asr: {tx}]")


if __name__ == "__main__":
    main()
