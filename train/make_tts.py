"""Synthetic training speech: the train-split prompts from modern sources (news, Wikipedia,
your terms, problem words) read by random Piper voices at random speeds. Teaches the
vocabulary; noise and room effects are added at training time.

    uv run make_tts.py                 # 2 renditions per prompt -> data/train/tts/, tts.jsonl
    uv run make_tts.py --per-prompt 4 --sources terms hardwords

Test-split prompts are never rendered.
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "datagen"))
OUT = ROOT / "data/train/tts"
PROMPTS = ROOT / "data/field/prompts.jsonl"

_V = {}


def _init():
    from tts import Voices
    _V["v"] = Voices()
    _V["specs"] = _V["v"].specs(n_multi=60)


def _render(job):
    pid, text, k, seed = job
    rng = random.Random(seed)
    spec = rng.choice(_V["specs"])
    speed = round(rng.uniform(0.75, 1.15), 2)
    pcm = _V["v"].say(text, spec, speed=speed, noise=rng.uniform(0.4, 0.8))
    path = OUT / f"{pid}-{k}.wav"
    sf.write(path, pcm, 16000, subtype="PCM_16")
    return {"file": str(path.relative_to(ROOT)), "prompt_id": pid, "voice": f"{spec[0]}:{spec[1]}",
            "speed": speed, "secs": round(len(pcm) / 16000, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--sources", nargs="*", default=["news", "wikipedia", "terms", "hardwords"])
    ap.add_argument("-j", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT.parent / "tts.jsonl"
    have = {json.loads(l)["file"] for l in open(index)} if index.exists() else set()
    prompts = [json.loads(l) for l in open(PROMPTS)]
    jobs = []
    for p in prompts:
        if p["split"] != "train" or p["source"] not in a.sources:
            continue
        for k in range(a.per_prompt):
            if f"data/train/tts/{p['id']}-{k}.wav" not in have:
                jobs.append((p["id"], p["display"], k, f"{p['id']}-{k}"))
    print(f"rendering {len(jobs)} clips ({a.j} processes)")
    ref = {p["id"]: p["ref"] for p in prompts}
    with mp.get_context("spawn").Pool(a.j, initializer=_init) as pool, open(index, "a") as f:
        for i, row in enumerate(pool.imap_unordered(_render, jobs, chunksize=4)):
            row["ref"] = ref[row["prompt_id"]]
            f.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    rows = [json.loads(l) for l in open(index)]
    print(f"{len(rows)} clips, {sum(r['secs'] for r in rows) / 3600:.2f} h -> {index.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
