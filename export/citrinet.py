"""Citrinet (NeMo ConvASREncoder + ConvASRDecoder) reimplemented in plain PyTorch.

Mirrors nemo/collections/asr/parts/submodules/jasper.py (JasperBlock, MaskedConv1d,
SqueezeExcite) and nemo/collections/asr/parts/preprocessing/features.py
(FilterbankFeatures) closely enough to load NeMo's state_dict by name and reproduce
its outputs. No NeMo dependency.

Layout: features [B, 80, T] (NCW), encoder downsamples time 8x, logits [B, T', 257]
(blank = last index, 256).
"""
from __future__ import annotations

import math
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_DIR = DATA / "models" / "citrinet_256_ls"
LIBRI = DATA / "librispeech" / "LibriSpeech"


# ----------------------------------------------------------------------------- model
def _mask(x: torch.Tensor, lens: torch.Tensor | None) -> torch.Tensor:
    if lens is None:
        return x
    t = torch.arange(x.shape[-1], device=x.device)[None, :]
    return x.masked_fill((t >= lens[:, None])[:, None, :], 0.0)


def _conv_len(lens, k, s, p):
    if lens is None:
        return None
    return torch.div(lens + 2 * p - (k - 1) - 1, s, rounding_mode="trunc") + 1


class SepConvBN(nn.Module):
    """depthwise conv(k, stride) -> pointwise 1x1 -> BatchNorm (NeMo separable conv_bn)."""

    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.k, self.stride, self.pad = k, stride, (k - 1) // 2
        self.dw = nn.Conv1d(cin, cin, k, stride=stride, padding=self.pad, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout, eps=1e-3)

    def forward(self, x, lens=None):
        x = self.dw(_mask(x, lens))
        lens = _conv_len(lens, self.k, self.stride, self.pad)
        x = self.pw(_mask(x, lens))
        return self.bn(x), lens


class ConvBN(nn.Module):
    """plain conv (residual 1x1 branch) -> BatchNorm."""

    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.k, self.stride, self.pad = k, stride, (k - 1) // 2
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, padding=self.pad, bias=False)
        self.bn = nn.BatchNorm1d(cout, eps=1e-3)

    def forward(self, x, lens=None):
        return self.bn(self.conv(_mask(x, lens))), _conv_len(lens, self.k, self.stride, self.pad)


class SqueezeExcite(nn.Module):
    """global (se_context_size=-1) squeeze-excitation: x * sigmoid(W2 relu(W1 mean_t(x)))."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction, bias=False)
        self.fc2 = nn.Linear(ch // reduction, ch, bias=False)

    def forward(self, x, lens=None):
        if lens is None:
            y = x.mean(dim=-1)  # static graph: global mean over the whole (padded) window
        else:
            x = _mask(x, lens)
            y = x.sum(dim=-1) / lens[:, None].to(x.dtype)
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y))))
        return x * y[:, :, None]


class JasperBlock(nn.Module):
    def __init__(self, cin, cfg):
        super().__init__()
        cout, rep = cfg["filters"], cfg["repeat"]
        k, s = cfg["kernel"][0], cfg["stride"][0]
        assert cfg["dilation"][0] == 1 and cfg["separable"] and cfg["se"] and cfg["se_context_size"] == -1
        assert cfg.get("dropout", 0.0) == 0.0
        stride_last = cfg.get("stride_last", False)
        self.convs = nn.ModuleList()
        c = cin
        for i in range(rep):
            last = i == rep - 1
            self.convs.append(SepConvBN(c, cout, k, s if (last or not stride_last) else 1))
            c = cout
        self.se = SqueezeExcite(cout)
        self.res = None
        if cfg.get("residual", False):
            rs = s if cfg.get("residual_mode", "add") == "stride_add" else 1
            self.res = ConvBN(cin, cout, 1, rs)

    def forward(self, x, lens=None):
        out, olens = x, lens
        for i, c in enumerate(self.convs):
            out, olens = c(out, olens)
            if i < len(self.convs) - 1:
                out = F.relu(out)
        out = self.se(out, olens)
        if self.res is not None:
            r, _ = self.res(x, lens)
            out = out + r
        return F.relu(out), olens


class Citrinet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        enc = cfg["encoder"]
        c = enc["feat_in"]
        self.blocks = nn.ModuleList()
        for b in enc["jasper"]:
            self.blocks.append(JasperBlock(c, b))
            c = b["filters"]
        ncls = cfg["decoder"]["num_classes"] + 1  # + blank
        self.decoder = nn.Conv1d(c, ncls, 1, bias=True)

    def forward(self, feats, lens=None):
        """feats [B,80,T] -> logits [B,257,T/8] (pre-softmax), out lens."""
        x = feats
        for b in self.blocks:
            x, lens = b(x, lens)
        return self.decoder(x), lens


# ----------------------------------------------------------------------------- loading
def unpack_nemo(model_dir: Path = MODEL_DIR):
    nemo = model_dir / "stt_en_citrinet_256_ls.nemo"
    if not (model_dir / "model_weights.ckpt").exists():
        with tarfile.open(nemo) as t:
            t.extractall(model_dir)


def map_state_dict(sd: dict, cfg: dict) -> dict:
    """NeMo key names -> ours. Every NeMo tensor must be consumed."""
    out, used = {}, set()

    def take(src, dst):
        out[dst] = sd[src]
        used.add(src)

    def bn(src, dst):
        for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            take(f"{src}.{p}", f"{dst}.{p}")

    for bi, b in enumerate(cfg["encoder"]["jasper"]):
        pre = f"encoder.encoder.{bi}"
        idx = 0
        for r in range(b["repeat"]):
            take(f"{pre}.mconv.{idx}.conv.weight", f"blocks.{bi}.convs.{r}.dw.weight")
            take(f"{pre}.mconv.{idx + 1}.conv.weight", f"blocks.{bi}.convs.{r}.pw.weight")
            bn(f"{pre}.mconv.{idx + 2}", f"blocks.{bi}.convs.{r}.bn")
            idx += 3 if r == b["repeat"] - 1 else 5  # conv,conv,bn (+relu,dropout)
        take(f"{pre}.mconv.{idx}.fc.0.weight", f"blocks.{bi}.se.fc1.weight")
        take(f"{pre}.mconv.{idx}.fc.2.weight", f"blocks.{bi}.se.fc2.weight")
        if b.get("residual", False):
            take(f"{pre}.res.0.0.conv.weight", f"blocks.{bi}.res.conv.weight")
            bn(f"{pre}.res.0.1", f"blocks.{bi}.res.bn")
    take("decoder.decoder_layers.0.weight", "decoder.weight")
    take("decoder.decoder_layers.0.bias", "decoder.bias")
    # preprocessor buffers are consumed by the feature extractor, not the network
    used |= {"preprocessor.featurizer.window", "preprocessor.featurizer.fb"}
    unused = set(sd) - used
    assert not unused, f"unconsumed NeMo tensors: {sorted(unused)[:10]}"
    return out


def load_model(model_dir: Path = MODEL_DIR):
    unpack_nemo(model_dir)
    cfg = yaml.safe_load(open(model_dir / "model_config.yaml"))
    sd = torch.load(model_dir / "model_weights.ckpt", map_location="cpu", weights_only=False)
    m = Citrinet(cfg)
    missing, unexpected = m.load_state_dict(map_state_dict(sd, cfg), strict=True)
    assert not missing and not unexpected
    m.eval()
    feat = MelFeatures(cfg["preprocessor"], sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"])
    vocab = list(cfg["decoder"]["vocabulary"])
    return m, feat, vocab, cfg


# ----------------------------------------------------------------------------- BN folding
@torch.no_grad()
def fold_bn(model: Citrinet) -> Citrinet:
    """Return a copy where every BatchNorm is folded into the preceding (pointwise) conv,
    which gains a bias. BN modules become Identity."""
    import copy

    m = copy.deepcopy(model).eval()

    def fold(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        new = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size[0], stride=conv.stride,
                        padding=conv.padding, groups=conv.groups, bias=True)
        new.weight.copy_(conv.weight * s[:, None, None])
        b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        new.bias.copy_((b0 - bn.running_mean) * s + bn.bias)
        return new

    for blk in m.blocks:
        for c in blk.convs:
            c.pw, c.bn = fold(c.pw, c.bn), nn.Identity()
        if blk.res is not None:
            blk.res.conv, blk.res.bn = fold(blk.res.conv, blk.res.bn), nn.Identity()
    return m


# ----------------------------------------------------------------------------- features
class MelFeatures:
    """NeMo FilterbankFeatures at inference (dither off, preemph 0.97, |STFT|^2, slaney mel,
    log(x + 2^-24), per_feature normalization with unbiased std + 1e-5)."""

    def __init__(self, pcfg, window: torch.Tensor, fb: torch.Tensor, pad_mode="constant"):
        self.sr = pcfg["sample_rate"]
        self.win_length = int(pcfg["window_size"] * self.sr)
        self.hop = int(pcfg["window_stride"] * self.sr)
        self.n_fft = pcfg["n_fft"]
        self.window = window.float()
        self.fb = fb.float()[0]  # [80, 257]
        self.preemph = 0.97
        self.guard = 2.0 ** -24
        self.pad_mode = pad_mode
        assert pcfg["normalize"] == "per_feature" and pcfg["features"] == 80

    def seq_len(self, n_samples: int) -> int:
        return n_samples // self.hop + 1

    @torch.no_grad()
    def logmel(self, audio: torch.Tensor) -> torch.Tensor:
        """audio [N] float in [-1,1] -> un-normalized log mel [80, T]."""
        x = audio.float()
        x = torch.cat((x[:1], x[1:] - self.preemph * x[:-1]))
        s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length, center=True,
                       window=self.window, return_complex=True, pad_mode=self.pad_mode)
        p = s.real.pow(2) + s.imag.pow(2)  # sqrt then pow(2) in NeMo; identical up to rounding
        return torch.log(self.fb @ p + self.guard)

    @staticmethod
    def normalize(x: torch.Tensor):
        """per_feature over time. x [80,T] -> normalized, mean, std."""
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(((x - mean) ** 2).sum(dim=1, keepdim=True) / (x.shape[1] - 1)) + 1e-5
        return (x - mean) / std, mean, std

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.logmel(audio)
        x = x[:, : self.seq_len(audio.shape[0])]
        return self.normalize(x)[0]


# ----------------------------------------------------------------------------- decoding / data
def ctc_greedy(logits: torch.Tensor, vocab: list[str], n: int | None = None) -> str:
    """logits [257, T] -> text. WordPiece: tokens joined by spaces, '##' continues a word."""
    ids = logits[:, :n].argmax(dim=0).tolist() if n is not None else logits.argmax(dim=0).tolist()
    blank = len(vocab)
    toks, prev = [], None
    for i in ids:
        if i != prev and i != blank and not (vocab[i].startswith("[") and vocab[i].endswith("]")):
            toks.append(vocab[i])
        prev = i
    # The WordPiece tokenizer (BERT basic tokenizer) split punctuation off at training time, so
    # "don't" was trained as [don, ', t]: glue the apostrophe to both neighbours.
    text, glue = "", False
    for t in toks:
        if t.startswith("##"):
            text += t[2:]
        elif t == "'":
            text += t
            glue = True
            continue
        else:
            text += ("" if (glue or not text) else " ") + t
        glue = False
    return text.strip()


def load_split(split="test-clean"):
    """-> list of (utt_id, flac_path, transcript lowercased)."""
    out = []
    root = LIBRI / split
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in open(trans):
            uid, txt = line.strip().split(" ", 1)
            out.append((uid, trans.parent / f"{uid}.flac", txt.lower()))
    return out


def read_audio(path) -> torch.Tensor:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000
    return torch.from_numpy(a)


def normalize_text(s: str) -> str:
    import re

    s = s.lower()
    s = re.sub(r"[^a-z' ]", " ", s)
    return " ".join(s.split())
