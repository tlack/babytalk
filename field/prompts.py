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
                   "LASER", "RADAR", "SCUBA", "NAFTA", "ASEAN", "LED", "GIF"}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


# ---------------------------------------------------------------- normalization
def _number(m):
    s = m.group(0).replace(",", "")
    try:
        if re.fullmatch(r"(1[1-9]|20)\d\d", s):         # years: 2026 -> twenty twenty-six
            return num2words(int(s), to="year")
        if "." in s:
            return num2words(float(s))
        return num2words(int(s))
    except Exception:
        return s


def normalize(text):
    """-> (display, reference) or None if the text can't be read unambiguously."""
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
    t = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", _number, t)
    t = re.sub(r"\b([A-Z]{2,5})s?\b",
               lambda m: m.group(0) if m.group(1) in SPOKEN_ACRONYMS else " ".join(m.group(1)), t)
    t = t.replace("&", " and ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    ref = re.sub(r"[^a-z' ]", " ", t.lower())
    ref = re.sub(r"\s+", " ", ref.replace(" '", " ").replace("' ", " ")).strip()
    if not ref or re.search(r"\d", t):
        return None
    return t, ref


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
    """Single sentences, plus runs of consecutive ones to reach the long buckets."""
    out = list(sents)
    for i in range(len(sents)):
        run = sents[i]
        for j in range(i + 1, len(sents)):
            run = run + " " + sents[j]
            out.append(run)
            if len(run.split()) > 56:
                break
    return out


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
            if a.get("extract"):
                out.extend(chunks(sentences(a["extract"])))
    return out


def from_terms():
    return [l.strip() for l in open(TERMS) if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7, help="days of Wikipedia most-read")
    ap.add_argument("--no-web", action="store_true")
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
    today = datetime.date.today().isoformat()
    added = {}
    for src, texts in srcs.items():
        for raw in texts:
            n = normalize(raw)
            if not n:
                continue
            display, ref = n
            b = bucket_of(len(ref.split()))
            if b is None and src != "terms":
                continue
            pid = hashlib.sha1(ref.encode()).hexdigest()[:12]
            if pid in old or pid in added:
                continue
            added[pid] = {"id": pid, "display": display, "ref": ref, "source": src,
                          "bucket": b or 1, "words": len(ref.split()),
                          "split": "test" if int(pid, 16) % 5 == 0 else "train", "added": today}
    with open(OUT, "a") as f:
        for r in added.values():
            f.write(json.dumps(r) + "\n")
    allp = list(old.values()) + list(added.values())
    print(f"added {len(added)} prompts -> {OUT} ({len(allp)} total)")
    for src in sorted({r['source'] for r in allp}):
        rs = [r for r in allp if r["source"] == src]
        by = {s: sum(r["bucket"] == s for r in rs) for s, _, _ in BUCKETS}
        print(f"  {src:12s} {len(rs):5d}  by bucket {by}  test {sum(r['split'] == 'test' for r in rs)}")


if __name__ == "__main__":
    main()
