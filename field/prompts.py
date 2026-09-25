"""Build the pool of sentences to record: what you read aloud, and what the laptop's TTS
voices say, in capture sessions.

    uv run prompts.py                 # -> data/field/prompts.jsonl (merges with what's there)
    uv run prompts.py --days 14       # more recent Wikipedia "most read" days

Sources: LibriSpeech dev-clean transcripts (the model's home turf), recent news headlines
and summaries (RSS), recent Wikipedia most-read article summaries, and field/terms.txt
(your own words: names, commands, the vocabulary an audiobook model never heard).

Every prompt is normalized so that what is shown is exactly what is read and what the
transcript is scored against: numbers spelled out, acronyms spelled as letters, no
symbols. Each gets a length bucket (about 1 / 4 / 8 / 12 / 20 s read aloud) and a split:
test for ~20% of texts (by a hash of the text, so a text is never in both), else train.
"""
import argparse
import datetime
import hashlib
import html
import json
import re
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from num2words import num2words

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/field/prompts.jsonl"
LIBRI = ROOT / "data/librispeech/LibriSpeech/dev-clean"
TERMS = Path(__file__).resolve().parent / "terms.txt"
UA = "esp32-micromodels/0.1 (speech field prompts)"
FEEDS = ["https://feeds.bbci.co.uk/news/rss.xml", "https://feeds.npr.org/1001/rss.xml",
         "https://feeds.bbci.co.uk/news/technology/rss.xml", "https://feeds.npr.org/1019/rss.xml"]

# (bucket seconds, word range) at a normal reading pace of ~2.5 words/s
BUCKETS = [(1, 1, 4), (4, 7, 13), (8, 16, 24), (12, 26, 34), (20, 42, 56)]
SPOKEN_ACRONYMS = {"NASA", "NATO", "COVID", "OPEC", "UNESCO", "UNICEF", "FEMA", "SWAT",
                   "LASER", "RADAR", "SCUBA", "NAFTA", "ASEAN", "LED", "GIF",
                   "CUDA", "JSON", "YAML", "TOML", "WASM", "RAM", "ROM", "SIM", "PIN", "CAPTCHA", "OPENAI"}


def fetch(url, tries=5):
    """GET with a polite pace and backoff on HTTP 429 (Wikipedia rate-limits bursts)."""
    import time
    import urllib.error
    for k in range(tries):
        time.sleep(0.3)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(2 * 2 ** k)


# ---------------------------------------------------------------- normalization
def _number(m):
    raw = m.group(0)
    s = raw.replace(",", "")
    try:
        if "," not in raw and re.fullmatch(r"(1[1-9]|20)\d\d", s):   # years: 2026 -> twenty twenty-six
            return num2words(int(s), to="year")
        if "." in s:                                      # 7.05 -> seven point zero five
            whole, frac = s.split(".", 1)
            return num2words(int(whole)) + " point " + " ".join(num2words(int(c)) for c in frac)
        return num2words(int(s))
    except Exception:
        return s


TECH = re.compile(r"\b\d{4}s\b"                                  # decades: 1990s
                  r"|\b\d+(?:\.\d+){2,}\b"                          # versions: 7.0.1
                  r"|\b(?=\w*[A-Za-z])(?=\w*\d)[A-Za-z0-9]+\b")        # x86, ud2, 4K, H100


def spoken(tok):
    """Best-guess reading of a tech token; the prompt is marked 'loose' (scoring less certain)."""
    if re.fullmatch(r"\d{4}s", tok):
        w = num2words(int(tok[:-1]), to="year").split()
        w[-1] = w[-1][:-1] + "ies" if w[-1].endswith("y") else w[-1] + "s"
        return " ".join(w)
    if re.fullmatch(r"\d+(?:\.\d+)+", tok):
        return " point ".join(" ".join(num2words(int(c)) for c in part) if part.startswith("0") and len(part) > 1
                              else num2words(int(part)) for part in tok.split("."))
    out = []
    for run in re.findall(r"[A-Za-z]+|\d+", tok):
        if run.isdigit():
            out.append(num2words(int(run)))
        elif len(run) <= 3 or run.isupper():
            out.append(" ".join(run.lower()))
        else:
            out.append(run.lower())
    return " ".join(out)


def normalize(text):
    """-> (display, reference, loose) or None if the text can't be used.

    display: what is shown and spoken. Numbers are spelled out; acronyms and tech tokens
    (x86, 4K, GPT-5, 7.0.1) stay as written, so people and TTS voices read them naturally.
    reference: what the transcript is scored against (acronyms and tech tokens spelled out).
    loose: True when a tech token's reading is a guess (x86 -> "x eighty six")."""
    t = html.unescape(text).replace("\u2013", " to ").replace("\u2014", ", ")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\s*\([^)]*\)", "", t)                 # parentheticals: dates, pronunciations
    t = t.replace('"', "")
    t = re.sub(r"\b(" + "|".join(SAY) + r")\.", lambda m: SAY[m.group(1)], t)
    if re.search(r"[@#_/\\|<>{}\[\]=+*~^]", t):
        return None
    t = re.sub(r"\$(\d[\d,.]*)\s*(million|billion|trillion)?",
               lambda m: m.group(1) + (" " + m.group(2) if m.group(2) else "") + " dollars", t)
    t = re.sub(r"(\d[\d,.]*)\s*%", r"\1 percent", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
    loose = bool(TECH.search(t))

    def expand(s_, keep_tech):
        s_ = TECH.sub(lambda m: m.group(0) if keep_tech else spoken(m.group(0)), s_)
        s_ = re.sub(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.]*\d)(?![A-Za-z])", _number, s_)
        s_ = s_.replace("&", " and ").replace("-", " ")
        return re.sub(r"\s+", " ", s_).strip()

    display, t = expand(t, True), expand(t, False)
    spelled = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", t)            # JetKVM -> Jet KVM
    spelled = re.sub(r"\b([A-Z]{2,5})s?\b",
                     lambda m: m.group(0) if m.group(1) in SPOKEN_ACRONYMS else " ".join(m.group(1)), spelled)
    ref = re.sub(r"[^a-z' ]", " ", spelled.lower())
    ref = re.sub(r"\s+", " ", ref.replace(" '", " ").replace("' ", " ")).strip()
    if not ref or re.search(r"\d", t):
        return None
    return display, ref, loose


SAY = {"Mr": "Mister", "Mrs": "Missus", "Dr": "Doctor", "Jr": "Junior", "Sr": "Senior",
       "Sen": "Senator", "Gov": "Governor", "Gen": "General", "Prof": "Professor", "Lt": "Lieutenant",
       "Capt": "Captain", "Col": "Colonel", "Mt": "Mount", "vs": "versus"}
ABBREV = {"Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "Gen", "Gov", "Sen", "Rep", "Lt", "Col", "Capt",
          "Prof", "Inc", "Co", "vs", "No", "Mt", "Ft"}


def sentences(text):
    """Split at . ! ? -- but not after initials (Robert F.), U.S. or Mr./Dr./St. and friends."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    out = []
    for p in parts:
        prev = out[-1] if out else ""
        last = re.search(r"(\S+)\.$", prev)
        if out and last and (len(last.group(1).replace(".", "")) <= 1 or last.group(1) in ABBREV
                             or re.fullmatch(r"(?:[A-Z]\.)+[A-Z]", last.group(1))):
            out[-1] = prev + " " + p
        else:
            out.append(p)
    return [s.strip() for s in out if s.strip()]


def bucket_of(n_words):
    for secs, lo, hi in BUCKETS:
        if lo <= n_words <= hi:
            return secs
    return None


def chunks(sents):
    """Single sentences, plus NON-overlapping runs of consecutive ones for the long buckets
    (overlapping runs made one article's sentences show up in many prompts)."""
    out = list(sents)
    run = ""
    for s_ in sents:
        run = (run + " " + s_).strip()
        if len(run.split()) >= 26:
            out.append(run)
            run = ""
    return out


def disambiguation(text):
    return bool(re.search(r"\bmay (also )?refer to\b|\brefers? to:", text, re.I))


# ---------------------------------------------------------------- sources
def from_librispeech(limit=600):
    rows = []
    for f in sorted(LIBRI.glob("*/*/*.trans.txt")):
        for line in open(f):
            uid, txt = line.strip().split(" ", 1)
            rows.append(txt.capitalize())
    step = max(1, len(rows) // limit)
    return rows[::step][:limit]


def from_news():
    out = []
    for url in FEEDS:
        try:
            root = ET.fromstring(fetch(url))
        except Exception as e:
            print(f"  news feed failed: {url}: {e}")
            continue
        for item in root.iter("item"):
            for tag in ("title", "description"):
                el = item.find(tag)
                if el is not None and el.text:
                    txt = re.sub(r"<[^>]+>", " ", el.text)
                    out.extend(chunks(sentences(txt)) if tag == "description" else [txt])
    return out


def from_wikipedia(days):
    out = []
    today = datetime.date.today()
    for d in range(1, days + 1):
        day = today - datetime.timedelta(days=d)
        try:
            feed = json.loads(fetch("https://api.wikimedia.org/feed/v1/wikipedia/en/featured/"
                                    + day.strftime("%Y/%m/%d")))
        except Exception as e:
            print(f"  wikipedia {day} failed: {e}")
            continue
        arts = feed.get("mostread", {}).get("articles", [])
        if feed.get("tfa"):
            arts.append(feed["tfa"])
        for a in arts:
            if a.get("extract") and a.get("type") != "disambiguation" and not disambiguation(a["extract"]):
                out.extend(chunks(sentences(a["extract"])))
    return out


def from_hardwords(top, per_word=4):
    """Sentences containing the words the model gets wrong (data/field/hard_words.json), from
    the intros of Wikipedia articles found by searching for each word."""
    hw = ROOT / "data/field/hard_words.json"
    if not hw.exists():
        print("  no hard-word list yet (export/hard_words.py)")
        return []
    words = [d["word"] for d in json.loads(hw.read_text())["words"]
             if d["kind"] == "vocab" and len(d["word"]) > 3][:top]
    out = []
    for w in words:
        try:
            q = urllib.parse.quote(w)
            res = json.loads(fetch("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
                                   f"&srlimit=6&srsearch={q}"))
            titles = [x["title"] for x in res["query"]["search"]]
            if not titles:
                continue
            ex = json.loads(fetch("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext"
                                  "&format=json&titles=" + urllib.parse.quote("|".join(titles))))
        except Exception as e:
            print(f"  wikipedia search failed for {w!r}: {e}")
            continue
        got = 0
        for page in ex["query"]["pages"].values():
            if disambiguation(page.get("extract", "")):
                continue
            for sent in sentences(page.get("extract", "")):
                if re.search(r"\b%s\b" % re.escape(w), sent, re.I) and got < per_word:
                    out.append(sent)
                    got += 1
    return out


HN_STAMP = ROOT / "data/field/.hn_last_request"


def hn_get(url):
    """One Hacker News request, at least 60 s after the previous one (even across runs)."""
    import time
    last = float(HN_STAMP.read_text()) if HN_STAMP.exists() else 0.0
    wait = 60 - (time.time() - last)
    if wait > 0:
        print(f"  (waiting {wait:.0f} s: at most one Hacker News request per minute)")
        time.sleep(wait)
    HN_STAMP.write_text(str(time.time()))
    return fetch(url).decode("utf-8", "replace")


def from_hackernews(n_days, done_days):
    """Titles from past HN front pages: random days 2-30 days ago, skipping days already fetched."""
    import random
    today = datetime.date.today()
    days = [today - datetime.timedelta(days=d) for d in range(2, 31)]
    days = [d for d in days if d.isoformat() not in done_days]
    random.shuffle(days)
    out = []
    for day in days[:n_days]:
        try:
            page = hn_get("https://news.ycombinator.com/front?day=" + day.isoformat())
        except Exception as e:
            print(f"  hacker news {day} failed: {e}")
            continue
        titles = re.findall(r'<span class="titleline"><a [^>]*>(.*?)</a>', page)
        for t in titles:
            t = re.sub(r"^(Show|Ask|Tell|Launch) HN:\s*", "", html.unescape(t))
            t = re.sub(r"\s*\[[^\]]*\]", "", t)          # [pdf], [video]
            out.append((t, day.isoformat()))
        print(f"  hacker news {day}: {len(titles)} titles")
    return out


def from_terms():
    return [l.strip() for l in open(TERMS) if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7, help="days of Wikipedia most-read")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--hn", type=int, default=2, metavar="DAYS",
                    help="Hacker News front-page days to add (random, 2-30 days ago; 1 request/min)")
    ap.add_argument("--hard", type=int, default=0, metavar="N",
                    help="also fetch sentences with the top N problem words (export/hard_words.py)")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            old[r["id"]] = r
    srcs = {"librispeech": from_librispeech(), "terms": from_terms()}
    if not a.no_web:
        srcs["news"] = from_news()
        srcs["wikipedia"] = from_wikipedia(a.days)
        if a.hard:
            srcs["hardwords"] = from_hardwords(a.hard)
        if a.hn:
            done_days = {r.get("hn_day") for r in old.values()}
            srcs["hackernews"] = from_hackernews(a.hn, done_days)
    today = datetime.date.today().isoformat()
    added = {}
    for src, texts in srcs.items():
        for raw in texts:
            extra = {}
            if isinstance(raw, tuple):                      # (text, hn_day)
                raw, extra = raw[0], {"hn_day": raw[1]}
            if disambiguation(raw):
                continue
            n = normalize(raw)
            if not n:
                continue
            display, ref, loose = n
            if loose:
                extra = {**extra, "loose": True}
            b = bucket_of(len(ref.split()))
            if b is None and src in ("terms", "hackernews"):   # keep every title / term
                nw = len(ref.split())
                b = min(BUCKETS, key=lambda x: min(abs(nw - x[1]), abs(nw - x[2])))[0]
            if b is None:
                continue
            pid = hashlib.sha1(ref.encode()).hexdigest()[:12]
            if pid in old or pid in added:
                continue
            added[pid] = {"id": pid, "display": display, "ref": ref, "source": src,
                          "bucket": b or 1, "words": len(ref.split()),
                          "split": "test" if int(pid, 16) % 5 == 0 else "train", "added": today, **extra}
    with open(OUT, "a") as f:
        for r in added.values():
            f.write(json.dumps(r) + "\n")
    allp = list(old.values()) + list(added.values())
    def unwanted(p):                  # older pools: disambiguation pages
        return disambiguation(p["display"])
    purged = [p for p in allp if unwanted(p)]
    if purged:
        allp = [p for p in allp if not unwanted(p)]
        with open(OUT, "w") as f:
            for r in allp:
                f.write(json.dumps(r) + "\n")
        print(f"  removed {len(purged)} disambiguation-page prompts")
    print(f"added {len(added)} prompts -> {OUT} ({len(allp)} total)")
    for src in sorted({r['source'] for r in allp}):
        rs = [r for r in allp if r["source"] == src]
        by = {s: sum(r["bucket"] == s for r in rs) for s, _, _ in BUCKETS}
        print(f"  {src:12s} {len(rs):5d}  by bucket {by}  test {sum(r['split'] == 'test' for r in rs)}")


if __name__ == "__main__":
    main()
