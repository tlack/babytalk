"""Export the BN-folded Citrinet-256 as a static-shape ONNX graph (opset 18) and verify it
against PyTorch with onnxruntime.

  uv run export_onnx.py [--win 1600]

Graph: feats [1,80,WIN] (normalized log-mel, NCW) -> logits [1,257,WIN/8] (pre-softmax;
argmax == argmax of log-softmax, so LogSoftmax is dropped). SE's two Linear layers are
exported as 1x1 Conv on the [1,C,1] pooled tensor so the whole net is Conv / Relu /
ReduceMean / Sigmoid / Mul / Add.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from citrinet import DATA, MODEL_DIR, fold_bn, load_model, load_split, read_audio
from evalwer import fixed_windows


class StaticSE(nn.Module):
    def __init__(self, se):
        super().__init__()
        c, r = se.fc1.in_features, se.fc1.out_features
        self.fc1 = nn.Conv1d(c, r, 1, bias=False)
        self.fc2 = nn.Conv1d(r, c, 1, bias=False)
        self.fc1.weight.data = se.fc1.weight.data[:, :, None].clone()
        self.fc2.weight.data = se.fc2.weight.data[:, :, None].clone()

    def forward(self, x):
        y = x.mean(dim=2, keepdim=True)  # ReduceMean over time -> [1,C,1]
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(y))))


DEAD_SE_TOL = 0.01


def se_is_dead(se) -> bool:
    """A trained-out SE: fc1/fc2 weights ~0, so fc2(relu(fc1(.))) ~ 0 and the gate is a
    constant sigmoid(0) = 0.5. Blocks 0, 10, 13, 14, 15 of this checkpoint (weights <= 8e-3,
    fc1 output measured exactly 0 on dev-clean). Their near-zero weights also break ESP-DL
    int8 (output_exp < in_exp + w_exp), so we fold the 0.5 into the preceding conv."""
    return se.fc1.weight.abs().max() < DEAD_SE_TOL and se.fc2.weight.abs().max() < DEAD_SE_TOL


class StaticBlock(nn.Module):
    def __init__(self, blk, prune_dead_se=True):
        super().__init__()
        self.convs = nn.ModuleList()
        for c in blk.convs:  # folded: dw (no bias) -> pw (bias)
            self.convs.append(nn.Sequential(c.dw, c.pw))
        if prune_dead_se and se_is_dead(blk.se):
            import copy

            last = copy.deepcopy(blk.convs[-1].pw)
            last.weight.data *= 0.5
            last.bias.data *= 0.5
            self.convs[-1] = nn.Sequential(blk.convs[-1].dw, last)
            self.se = nn.Identity()
        else:
            self.se = StaticSE(blk.se)
        self.res = blk.res.conv if blk.res is not None else None

    def forward(self, x):
        out = x
        for i, c in enumerate(self.convs):
            out = c(out)
            if i < len(self.convs) - 1:
                out = F.relu(out)
        out = self.se(out)
        if self.res is not None:
            out = out + self.res(x)
        return F.relu(out)


class StaticCitrinet(nn.Module):
    def __init__(self, folded, prune_dead_se=True):
        super().__init__()
        self.blocks = nn.Sequential(*[StaticBlock(b, prune_dead_se) for b in folded.blocks])
        self.decoder = folded.decoder

    def forward(self, feats):
        return self.decoder(self.blocks(feats))


def build(win=1600, prune_dead_se=True):
    m, feat, vocab, cfg = load_model()
    sm = StaticCitrinet(fold_bn(m), prune_dead_se).eval()
    print("dead SE pruned in blocks:", [i for i, b in enumerate(sm.blocks) if isinstance(b.se, nn.Identity)])
    # equivalence with the reference (unfolded, unmasked) model on random input
    x = torch.randn(1, 80, win)
    with torch.no_grad():
        d = (sm(x) - m(x)[0]).abs().max().item()
    assert d < 1e-3, d
    return sm, m, feat, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=1600)
    a = ap.parse_args()
    sm, m, feat, vocab = build(a.win)
    out = DATA / "models" / f"citrinet256_static{a.win}.onnx"
    x = torch.randn(1, 80, a.win)
    torch.onnx.export(sm, (x,), str(out), opset_version=18, input_names=["feats"], output_names=["logits"],
                      dynamo=False, do_constant_folding=True)
    g = onnx.load(str(out))
    onnx.checker.check_model(g)
    ops = Counter(n.op_type for n in g.graph.node)
    print("ops:", dict(ops))
    print("input", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.graph.input])
    print("output", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.graph.output])
    nparam = sum(int(np.prod(t.dims)) for t in g.graph.initializer)
    print(f"params in graph: {nparam}  file: {out.stat().st_size / 1e6:.1f} MB  -> {out}")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    agree = []
    with torch.no_grad():
        for uid, p, _ in load_split()[::131]:
            a_ = read_audio(p)
            lm = feat.logmel(a_)[:, : feat.seq_len(a_.shape[0])]
            xw, v = fixed_windows(lm, a.win, fill="tile16")[0]
            ref = sm(xw[None]).numpy()
            got = sess.run(None, {"feats": xw[None].numpy()})[0]
            worst = max(worst, float(np.abs(ref - got).max() / np.abs(ref).max()))
            agree.append((ref.argmax(1) == got.argmax(1)).mean())
    print(f"ORT vs torch: worst rel diff {worst:.2e}, argmax agreement {np.mean(agree):.5f}")
    assert worst < 1e-4


if __name__ == "__main__":
    main()
