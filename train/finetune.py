"""Fine-tune Citrinet-256 (float) for the field: LibriSpeech (keeps what it knows), field
recordings (train split), TTS renditions of modern prompts (vocabulary), with noise from
the captured noise beds and other damage mixed in on the fly.

    uv run finetune.py --steps 4000                  # -> data/train/runs/<stamp>/model.pt
    uv run finetune.py --steps 200 --eval-every 100  # quick smoke test
    uv run finetune.py --device cpu                  # no GPU: works, but ~1.5 days instead of ~15 min

--device: auto (the default: an NVIDIA GPU if there is one, else Apple's GPU, else the CPU),
cuda, mps or cpu.

Score the result on the held-out field recordings:
    cd ../export && uv run field_eval.py --float-ckpt ../data/train/runs/<stamp>/model.pt

Batch norm statistics stay frozen (small, mixed batches would drag them around); every
weight trains. Test-split recordings and prompts are never used.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export"))
from citrinet import ctc_greedy, load_model, normalize_text  # noqa: E402

FIELD = ROOT / "data/field"
LIBRI = ROOT / "data/librispeech/LibriSpeech"
RATE = 16000
MAX_SECS = 20.0


# ---------------------------------------------------------------- targets (WordPiece)
class WordPiece:
    """BERT-style: punctuation (the apostrophe) is its own token, then greedy longest-match
    pieces with ## continuations -- the tokenizer the model was trained with."""

    def __init__(self, vocab):
        self.id = {t: i for i, t in enumerate(vocab)}

    def encode(self, text):
        ids = []
        for word in normalize_text(text).split():
            for part in [p for p in word.replace("'", " ' ").split()]:
                if part == "'":
                    ids.append(self.id["'"])
                    continue
                i, first = 0, True
                while i < len(part):
                    for j in range(len(part), i, -1):
                        tok = (part[i:j] if first else "##" + part[i:j])
                        if tok in self.id:
                            ids.append(self.id[tok])
                            i, first = j, False
                            break
                    else:
                        i += 1          # no piece for this character: skip it
        return ids


# ---------------------------------------------------------------- data
def libri_items(split):
    out = []
    for f in sorted((LIBRI / split).glob("*/*/*.trans.txt")):
        for line in open(f):
            uid, txt = line.strip().split(" ", 1)
            out.append((str(f.parent / f"{uid}.flac"), txt.lower(), None))
    return out


def field_items():
    out = []
    for l in open(FIELD / "manifest.jsonl"):
        r = json.loads(l)
        if r["split"] == "train" and not r.get("exclude"):
            out.append((str(FIELD / r["file"]), r["ref"], "stereo"))
    return out


def tts_items():
    p = ROOT / "data/train/tts.jsonl"
    return [(str(ROOT / r["file"]), r["ref"], None) for r in map(json.loads, open(p))] if p.exists() else []


def load_noise():
    """All captured noise (beds + session floors), both mics, as one list of float arrays."""
    beds = []
    for l in open(FIELD / "noise.jsonl"):
        r = json.loads(l)
        x, _ = sf.read(FIELD / r["file"], dtype="float32")
        for ch in range(x.shape[1] if x.ndim == 2 else 1):
            beds.append(x[:, ch] if x.ndim == 2 else x)
    return beds


def read(path, stereo, rng):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x[:, rng.randrange(x.shape[1])] if stereo else x[:, 0]   # field: either mic
    return x if sr == RATE else soxr.resample(x, sr, RATE)


def rms(x):
    return float(np.sqrt(np.mean(x * x)) + 1e-9)


def augment(x, kind, noise, rng):
    """Damage in roughly the order the world applies it."""
    if rng.random() < 0.3:                                        # speaking rate
        x = soxr.resample(x, RATE, RATE / rng.uniform(0.9, 1.1)).astype(np.float32)
    p_noise = 0.3 if kind == "field" else 0.75                    # field clips are already noisy
    if noise and rng.random() < p_noise:
        n = noise[rng.randrange(len(noise))]
        if len(n) < len(x):
            n = np.tile(n, len(x) // len(n) + 1)
        o = rng.randrange(len(n) - len(x) + 1)
        n = n[o:o + len(x)]
        snr = rng.uniform(-3, 22) if kind != "field" else rng.uniform(5, 25)
        x = x + n * (rms(x) / rms(n)) * 10 ** (-snr / 20)
    if rng.random() < 0.15:                                       # muffled (pocket, cloth)
        cut = rng.uniform(1500, 4000)
        X = np.fft.rfft(x)
        X[int(cut / (RATE / 2) * len(X)):] *= 0.05
        x = np.fft.irfft(X, len(x)).astype(np.float32)
    if rng.random() < 0.5:                                        # level, and clipping when loud
        x = np.clip(x * 10 ** (rng.uniform(-15, 12) / 20), -1, 1)
    if rng.random() < 0.1:                                        # bit-crushed (16 -> 4..8 bits)
        q = 2 ** (rng.randint(4, 8) - 1)
        x = np.round(x * q) / q
    if rng.random() < 0.1:                                        # dropouts
        for _ in range(rng.randint(1, 4)):
            w = int(rng.uniform(0.02, 0.12) * RATE)
            o = rng.randrange(max(1, len(x) - w))
            x[o:o + w] = 0
    return x.astype(np.float32)


class Mix(torch.utils.data.IterableDataset):
    def __init__(self, sources, weights, noise, feat, wp, seed):
        self.sources, self.weights, self.noise, self.feat, self.wp, self.seed = \
            sources, weights, noise, feat, wp, seed

    def __iter__(self):
        torch.set_num_threads(1)
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 7919)
        names = list(self.sources)
        while True:
            kind = rng.choices(names, weights=[self.weights[n] for n in names])[0]
            path, ref, stereo = rng.choice(self.sources[kind])
            try:
                x = read(path, stereo, rng)
            except Exception:
                continue
            if len(x) > MAX_SECS * RATE or len(x) < 0.3 * RATE:
                continue
            ids = self.wp.encode(ref)
            if not ids:
                continue
            x = augment(x, kind, self.noise, rng)
            f = self.feat(torch.from_numpy(x))                    # [80, T], normalized per clip
            if f.shape[1] // 8 < len(ids):                        # CTC needs a frame per token
                continue
            yield f, torch.tensor(ids), kind


def collate(batch):
    T = max(f.shape[1] for f, _, _ in batch)
    x = torch.zeros(len(batch), 80, T)
    for i, (f, _, _) in enumerate(batch):
        x[i, :, :f.shape[1]] = f
    lens = torch.tensor([f.shape[1] for f, _, _ in batch])
    tg = torch.cat([t for _, t, _ in batch])
    tl = torch.tensor([len(t) for _, t, _ in batch])
    return x, lens, tg, tl, [k for _, _, k in batch]


def spec_augment(x, lens, rng):
    for i in range(x.shape[0]):
        L = int(lens[i])
        for _ in range(2):
            w = rng.randint(0, 15)
            f0 = rng.randint(0, 80 - w)
            x[i, f0:f0 + w, :L] = 0
        for _ in range(2):
            w = rng.randint(0, max(1, min(40, L // 20)))
            t0 = rng.randint(0, max(0, L - w))
            x[i, :, t0:t0 + w] = 0
    return x


# ---------------------------------------------------------------- eval (float, whole clips)
@torch.no_grad()
def wer_on(m, feat, vocab, items, dev):
    m.eval()
    refs, hyps = [], []
    for path, ref, _ in items:
        x, sr = sf.read(path, dtype="float32")
        x = x[:, 0] if x.ndim == 2 else x
        f = feat(torch.from_numpy(x))[None].to(dev)
        lg, _ = m(f)
        refs.append(normalize_text(ref))
        hyps.append(normalize_text(ctc_greedy(lg[0].float().cpu(), vocab)))
    freeze_bn_train(m)
    keep = [i for i, r in enumerate(refs) if r]
    return 100 * jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep])


def freeze_bn_train(m):
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weights", default="libri=0.5,tts=0.3,field=0.2", help="sampling mix")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500, help="resumable checkpoint interval (steps)")
    ap.add_argument("--resume", help="a run directory to continue from its last checkpoint (ckpt.pt)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="where to train (auto: cuda, else mps, else cpu)")
    a = ap.parse_args()

    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", dev)
    torch.manual_seed(a.seed)
    if dev != "cpu":
        torch.set_num_threads(2)        # CPU work is small next to a GPU; all cores would thrash
    # (on the CPU, torch's default -- all cores -- trains; the data workers share them)
    ctc_dev = "cpu" if dev == "mps" else dev   # PyTorch has no CTC loss on Apple's GPU
    m, feat, vocab, cfg = load_model()
    m.to(dev)
    wp = WordPiece(vocab)
    sources = {"libri": libri_items("train-clean-100"), "tts": tts_items(), "field": field_items()}
    weights = dict(kv.split("=") for kv in a.weights.split(","))
    weights = {k: float(v) for k, v in weights.items() if sources.get(k)}
    sources = {k: v for k, v in sources.items() if k in weights}
    noise = load_noise()
    print("data:", {k: len(v) for k, v in sources.items()}, "weights", weights,
          f"noise {sum(len(n) for n in noise) / RATE / 60:.1f} min")

    # held-out checks: the field test split and a LibriSpeech dev-clean slice (forgetting)
    field_test = [(str(FIELD / r["file"]), r["ref"], None) for r in map(json.loads, open(FIELD / "manifest.jsonl"))
                  if r["split"] == "test" and not r.get("exclude")]
    dev_clean = libri_items("dev-clean")[::27][:100]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path(a.resume) if a.resume else ROOT / "data/train/runs" / stamp
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({**vars(a), "data": {k: len(v) for k, v in sources.items()}}, indent=1))
    log = open(run / "log.jsonl", "a")

    def evaluate(step):
        r = {"step": step, "field_test": round(wer_on(m, feat, vocab, field_test, dev), 2),
             "dev_clean": round(wer_on(m, feat, vocab, dev_clean, dev), 2)}
        print(f"  eval step {step}: field test {r['field_test']}% ({len(field_test)} clips), "
              f"dev-clean {r['dev_clean']}% ({len(dev_clean)})", flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()

    start = 0
    ck = None
    if a.resume and (run / "ckpt.pt").exists():
        ck = torch.load(run / "ckpt.pt", map_location="cpu")
        m.load_state_dict(ck["model"])
        start = ck["step"]
        print(f"resuming {run.name} at step {start}")
    freeze_bn_train(m)
    if not start:
        evaluate(0)
    dl = torch.utils.data.DataLoader(Mix(sources, weights, noise, feat, wp, a.seed), batch_size=a.batch,
                                     collate_fn=collate, num_workers=a.workers, prefetch_factor=4)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-3)
    warm = min(300, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    if ck:
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
    rng = random.Random(a.seed + start)
    t0, tl_sum, n_sum = time.time(), 0.0, 0
    for step, (x, lens, tg, tl, kinds) in enumerate(dl, start + 1):
        x = spec_augment(x, lens, rng).to(dev, non_blocking=True)
        lg, olens = m(x, lens.to(dev))      # fp32: bf16 autocast measured slower on this model
        lp = F.log_softmax(lg.float(), dim=1).permute(2, 0, 1)
        loss = F.ctc_loss(lp.to(ctc_dev), tg.to(ctc_dev), olens.to(ctc_dev), tl.to(ctc_dev), blank=len(vocab),
                          zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        tl_sum += loss.item()
        n_sum += 1
        if step % 50 == 0:
            el = time.time() - t0
            rate = (step - start) / el
            print(f"step {step}/{a.steps}  loss {tl_sum / n_sum:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{rate:.2f} steps/s  eta {(a.steps - step) / rate / 60:.0f} min", flush=True)
            log.write(json.dumps({"step": step, "loss": tl_sum / n_sum}) + "\n")
            tl_sum, n_sum = 0.0, 0
        if step % a.ckpt_every == 0:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, run / "ckpt.pt")
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, run / "model.pt")
        if step >= a.steps:
            break
    print(f"done -> {run.relative_to(ROOT)}/model.pt")


if __name__ == "__main__":
    main()
