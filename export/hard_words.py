"""The running list of words the models get wrong on the field recordings.

    uv run hard_words.py              # -> data/field/hard_words.json, top of the list printed
    uv run hard_words.py --top 60

Reads every field_eval_*.tsv in data/results (the newest transcript of each clip wins),
aligns reference and transcript word by word, and counts per reference word how often it
was heard and how often each model missed it (substituted or dropped), with what it was
heard as. A word the float model misses is a training/vocabulary problem; one only int8
or int4 miss is a quantization problem.

field/capture.py --hard records prompts containing these words, --retake re-records the
prompts that had errors, and field/prompts.py --hard fetches new sentences with them.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import jiwer

from field_eval import norm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data/results"
OUT = ROOT / "data/field/hard_words.json"
# too common to be interesting on their own: misses there are mostly alignment noise
STOP = set("a an the and or but of to in on at for with by from as is was are were be been it its this that "
           "he she they we you i his her their our my your him them us me".split())


def latest_transcripts():
    """clip id -> {"ref", "who", "condition", model: hyp}; newer runs override older."""
    clips = {}
    for tsv in sorted(RES.glob("field_eval_*.tsv")):
        lines = tsv.read_text().splitlines()
        head = lines[0].split("\t")
        for line in lines[1:]:
            cells = line.split("\t")
            row = dict(zip(head, cells))
            clips.setdefault(row["id"], {}).update(row)
    return clips


def misses(ref, hyp):
    """-> list of (ref word, heard as or '' for dropped) for every ref word not matched."""
    out = []
    if not ref.strip():
        return out
    o = jiwer.process_words(norm(ref), norm(hyp) if hyp.strip() else "<none>")
    r, h = o.references[0], o.hypotheses[0]
    for c in o.alignments[0]:
        if c.type == "substitute":
            for k in range(c.ref_end_idx - c.ref_start_idx):
                j = c.hyp_start_idx + k
                out.append((r[c.ref_start_idx + k], h[j] if j < c.hyp_end_idx else ""))
        elif c.type == "delete":
            out += [(w, "") for w in r[c.ref_start_idx:c.ref_end_idx]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    clips = latest_transcripts()
    models = [m for m in ("float", "int8", "int4") if any(m in c for c in clips.values())]
    words = collections.defaultdict(lambda: {"seen": 0, "clips": [], **{f"miss_{m}": 0 for m in models},
                                             "heard_as": collections.Counter()})
    clip_errors = {}
    for cid, c in clips.items():
        ref = norm(c.get("ref", ""))
        for w in ref.split():
            words[w]["seen"] += 1
        for m in models:
            ms = misses(ref, c.get(m, ""))
            if m == "float":
                clip_errors[cid] = len(ms)
            for w, heard in ms:
                d = words[w]
                d[f"miss_{m}"] += 1
                if m == "float":
                    d["heard_as"][heard or "(dropped)"] += 1
                    if cid not in d["clips"]:
                        d["clips"].append(cid)
    table = []
    for w, d in words.items():
        if w in STOP or not any(d[f"miss_{m}"] for m in models):
            continue
        d["heard_as"] = dict(d["heard_as"].most_common(5))
        d["word"] = w
        d["kind"] = "vocab" if d.get("miss_float") else "quant"
        table.append(d)
    table.sort(key=lambda d: (-d.get("miss_float", 0), -sum(d[f"miss_{m}"] for m in models), d["word"]))
    OUT.write_text(json.dumps({"models": models, "clips": len(clips), "clip_float_errors": clip_errors,
                               "words": table}, indent=1))
    print(f"{len(table)} problem words over {len(clips)} clips -> {OUT.relative_to(ROOT)}\n")
    print(f"{'word':18s} {'seen':>4s} " + " ".join(f"{m:>5s}" for m in models) + "  heard as (float)")
    for d in table[:a.top]:
        print(f"{d['word']:18s} {d['seen']:4d} " + " ".join(f"{d[f'miss_{m}']:5d}" for m in models)
              + "  " + ", ".join(f"{k} x{v}" for k, v in d["heard_as"].items()))
    q = [d for d in table if d["kind"] == "quant"]
    print(f"\n{len(table) - len(q)} words missed by the float model (vocabulary / training), "
          f"{len(q)} only by int8/int4 (quantization)")


if __name__ == "__main__":
    main()
