"""Verify BatchNorm folding: folded vs unfolded logits on real utterances (masked + static paths)."""
import torch

from citrinet import fold_bn, load_model, load_split, read_audio

torch.manual_seed(0)
m, feat, vocab, _ = load_model()
mf = fold_bn(m)
assert not any(isinstance(x, torch.nn.BatchNorm1d) for x in mf.modules())
worst = 0.0
with torch.no_grad():
    for uid, p, _ in load_split()[::200]:
        x = feat(read_audio(p))[None]
        a, _ = m(x)
        b, _ = mf(x)
        d = (a - b).abs().max().item() / a.abs().max().item()
        worst = max(worst, d)
        print(f"{uid}  T={x.shape[-1]:5d}  max|diff|={(a - b).abs().max().item():.2e}  rel={d:.2e}  argmax agree={(a.argmax(1) == b.argmax(1)).float().mean().item():.4f}")
print(f"worst relative diff {worst:.2e}")
assert worst < 1e-4
