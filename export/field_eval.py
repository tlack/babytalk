"""Score speech models on the field recordings (field/capture.py): word error rate overall
and by condition, speaker (you vs the laptop's TTS voices), clip length and text source.

    uv run field_eval.py                             # test split, float + int8 + int4
    uv run field_eval.py --split all --session 20260924-150102-desk-quiet
    uv run field_eval.py --mmrt ../path/finetuned_int4.mmrt --float-ckpt ../path/finetuned.pt
    uv run field_eval.py --legacy                    # also the older data/recordings clips

.mmrt models run through the host C executor (bit-exact with the board), on each whole clip
in one pass like the board's stt engine. Results: a table on stdout, per-clip transcripts in
data/results/field_eval_<stamp>.tsv and the numbers in data/results/field_eval.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import datetime
import json
import multiprocessing as mp
from pathlib import Path

import jiwer
import numpy as np
import torch

from citrinet import ctc_greedy, load_model, normalize_text, read_audio
from mmrt_check_model import ALLOC, FREE, LIB, build

ROOT = Path(__file__).resolve().parent.parent
FIELD = ROOT / "data/field"
RES = ROOT / "data/results"
MODELS = {"int8": ROOT / "models/citrinet256_int8.mmrt", "int4": ROOT / "models/citrinet256_int4.mmrt"}


def load_items(a):
    rows = []
    if (FIELD / "manifest.jsonl").exists():
        for l in open(FIELD / "manifest.jsonl"):
            r = json.loads(l)
            if r.get("exclude"):       # e.g. audio known not to match its reference
                continue
            if a.split != "all" and r["split"] != a.split:
                continue
            if a.session and r["session"] not in a.session:
                continue
            if a.condition and r["condition"] not in a.condition:
                continue
            r["path"] = str(FIELD / r["file"])
            r["who"] = "human" if r["speaker"] == "human" else "tts"
            rows.append(r)
    if a.legacy:
        from recordings import load_recordings
        for uid, path, ref in load_recordings():
            rows.append({"id": uid, "path": str(path), "ref": ref, "condition": "legacy-desk",
                         "who": "human" if uid.startswith("me-") else "replay", "bucket": 0, "source": "legacy"})
    return rows


# ---------------------------------------------------------------- mmrt (host C executor)
_W = {}


def _init(path):
    lib = ctypes.CDLL(str(LIB))
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    img = open(path, "rb").read()
    buf = ctypes.create_string_buffer(img, len(img))
    model = ctypes.create_string_buffer(256)
    assert lib.mmrt_open(model, buf, len(img)) == 0, f"not an .mmrt model: {path}"
    lib.mmrt_run.restype = ctypes.POINTER(ctypes.c_int8)
    lib.mmrt_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int), ALLOC, FREE]
    _, feat, vocab, _ = load_model()
    _W.update(lib=lib, buf=buf, model=model, feat=feat, vocab=vocab,
              alloc=ALLOC(lambda n: libc.malloc(n)), free=FREE(lambda p: libc.free(p)))


def _mmrt_one(path):
    """Whole clip, one pass: features normalized over the clip, input exponent -5."""
    from citrinet import MelFeatures
    feat, lib = _W["feat"], _W["lib"]
    audio = read_audio(path)
    lm = feat.logmel(audio)[:, : feat.seq_len(audio.shape[0])]
    x = MelFeatures.normalize(lm)[0]
    q = np.ascontiguousarray(np.clip(np.round(x.numpy().T * 32), -128, 127).astype(np.int8))
    T_out = ctypes.c_int()
    y = lib.mmrt_run(_W["model"], q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), q.shape[0],
                     ctypes.byref(T_out), _W["alloc"], _W["free"])
    assert y
    lg = np.ctypeslib.as_array(y, shape=(T_out.value * 272,)).reshape(T_out.value, 272)[:, :257]
    return ctc_greedy(torch.from_numpy(lg.T.astype(np.float32)), _W["vocab"])


def run_mmrt(path, items, jobs):
    # spawn, not fork: forking after PyTorch has started its thread pool deadlocks the workers
    with mp.get_context("spawn").Pool(jobs, initializer=_init, initargs=(str(path),)) as pool:
        return pool.map(_mmrt_one, [r["path"] for r in items], chunksize=2)


# ---------------------------------------------------------------- float (PyTorch)
@torch.no_grad()
def run_float(items, ckpt=None):
    m, feat, vocab, _ = load_model()
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    out = []
    for r in items:
        audio = read_audio(r["path"])
        lg, _ = m(feat(audio)[None])
        out.append(ctc_greedy(lg[0], vocab))
    return out


# ---------------------------------------------------------------- report
# The model learned from audiobooks with British spellings; don't count them as errors.
UK_US = {"colour": "color", "colours": "colors", "centre": "center", "centres": "centers",
         "favour": "favor", "favourite": "favorite", "honour": "honor", "honours": "honors",
         "behaviour": "behavior", "labour": "labor", "neighbour": "neighbor", "neighbours": "neighbors",
         "harbour": "harbor", "humour": "humor", "rumour": "rumor", "vapour": "vapor", "armour": "armor",
         "theatre": "theater", "metre": "meter", "metres": "meters", "litre": "liter", "fibre": "fiber",
         "defence": "defense", "offence": "offense", "licence": "license", "grey": "gray",
         "programme": "program", "travelled": "traveled", "travelling": "traveling", "jewellery": "jewelry",
         "cheque": "check", "tyre": "tire", "tyres": "tires", "realise": "realize", "realised": "realized",
         "recognise": "recognize", "recognised": "recognized", "organise": "organize", "organised": "organized",
         "apologise": "apologize", "analyse": "analyze", "catalogue": "catalog", "dialogue": "dialog"}


def norm(t):
    return " ".join(UK_US.get(w, w) for w in normalize_text(t).split())


def wer(pairs):
    refs = [norm(r) for r, _ in pairs]
    hyps = [norm(h) for _, h in pairs]
    keep = [i for i, r in enumerate(refs) if r]
    if not keep:
        return None
    return 100 * jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--session", nargs="*", help="only these sessions")
    ap.add_argument("--condition", nargs="*", help="only these conditions")
    ap.add_argument("--legacy", action="store_true", help="also score data/recordings")
    ap.add_argument("--models", nargs="*", default=["float", "int8", "int4"])
    ap.add_argument("--mmrt", nargs="*", default=[], help="extra .mmrt files to score")
    ap.add_argument("--float-ckpt", help="fine-tuned float weights (state_dict) to score as 'float-ft'")
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args()

    items = load_items(a)
    if not items:
        raise SystemExit("no recordings match (capture some with field/capture.py, or try --split all)")
    build()
    hyps = {}
    if "float" in a.models:
        hyps["float"] = run_float(items)
    if a.float_ckpt:
        hyps["float-ft"] = run_float(items, a.float_ckpt)
    for name in [m for m in a.models if m in MODELS]:
        hyps[name] = run_mmrt(MODELS[name], items, a.j)
    for p in a.mmrt:
        hyps[Path(p).stem] = run_mmrt(Path(p), items, a.j)

    groups = collections.OrderedDict()
    groups["ALL"] = list(range(len(items)))
    for key, fmt in (("condition", "{}"), ("who", "speaker {}"), ("bucket", "~{} s clips"), ("source", "{} text")):
        for i, r in enumerate(items):
            groups.setdefault(fmt.format(r.get(key)), []).append(i)
    names = list(hyps)
    print(f"\nword error rate (%), {len(items)} clips, split={a.split}\n")
    print(f"| group | clips | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    table = {}
    for g, idx in groups.items():
        row = {n: wer([(items[i]["ref"], hyps[n][i]) for i in idx]) for n in names}
        table[g] = {"clips": len(idx), **row}
        cells = " | ".join("-" if v is None else f"{v:.1f}" for v in row.values())
        print(f"| {g} | {len(idx)} | {cells} |")

    RES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RES / f"field_eval_{stamp}.tsv", "w") as f:
        f.write("id\tcondition\twho\tref\t" + "\t".join(names) + "\n")
        for i, r in enumerate(items):
            f.write(f"{r['id']}\t{r.get('condition')}\t{r.get('who')}\t{normalize_text(r['ref'])}\t"
                    + "\t".join(normalize_text(hyps[n][i]) for n in names) + "\n")
    with open(RES / "field_eval.jsonl", "a") as f:
        f.write(json.dumps({"stamp": stamp, "split": a.split, "clips": len(items), "table": table,
                            "sessions": sorted({r.get('session', 'legacy') for r in items})}) + "\n")
    print(f"\nper-clip transcripts: {(RES / f'field_eval_{stamp}.tsv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
