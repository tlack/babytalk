# Speech models

Ready-to-flash speech-to-text models for the runtime in `mmrt/`. All four are NVIDIA's
[Citrinet-256](https://huggingface.co/nvidia/stt_en_citrinet_256_ls) (English, trained on
LibriSpeech), converted and quantized by this project's `export/` scripts. Each comes in a
4-bit and an 8-bit version:

- **The original** (`citrinet256_int4/int8`): NVIDIA's weights. The default, and the best on
  clean, close-up speech.
- **Noisy rooms** (`citrinet256_noisy_int4/int8`): fine-tuned on recordings made on the
  boards with music, traffic and other noise in the background. **An alternative, not an
  upgrade:** it gets a quarter to a third fewer words wrong with noise in the room, and
  more wrong on clean speech.

| file | weights | size | noisy rooms* | clean speech** | fits |
|---|---|---|---|---|---|
| `citrinet256_int4.mmrt` | 4-bit (first and last layers 8-bit) | 5.99 MB | 58.1% | 8.2% | the standard 6 MB `model` partition |
| `citrinet256_int8.mmrt` | 8-bit | 9.78 MB | 51.5% | 6.3% | the int8 layout's 10 MB partition |
| `citrinet256_noisy_int4.mmrt` | 4-bit (first and last layers 8-bit) | 5.99 MB | **42.7%** | 11.8% | the standard 6 MB `model` partition |
| `citrinet256_noisy_int8.mmrt` | 8-bit | 9.78 MB | **33.8%** | 7.8% | the int8 layout's 10 MB partition |

Word error rate, lower is better.
\*82 held-out field recordings (`export/field_eval.py`): sentences read by synthetic voices
from a laptop speaker over music, highway noise, a coffee-shop loop and other background
sound, recorded by the board's mic. None of them were used in training, but they come from
the same rooms and voices as the training recordings, so expect a smaller gain in a new
setting.
\*\*LibriSpeech test-clean, 655 utterances. The original float model: 3.8%.

The model's format is recorded in the file, so any of the four flashes the same way, with no
firmware change (how: [`docs/MODELS.md`](../docs/MODELS.md#1-our-speech-models)).

SHA-256:

```
bdb6be064984b86af7c7ead182b2852420edaae252733264cd76f81bad0f8d47  citrinet256_int4.mmrt
6fd16badafdb2d02358682faec3055de93e011958d6b1431222ef27c66b8b493  citrinet256_int8.mmrt
39a875c9a494efa1afc51d6be7c34564d8339ffc9752b5100d830312c5995edf  citrinet256_noisy_int4.mmrt
ddc3780ebabd4aae479c0809c23d96100243afe28ef2e85413c35fe952ba1c60  citrinet256_noisy_int8.mmrt
```

**How the noisy-room models were made** (2026-10-09): 4,000 steps of `train/finetune.py`
from NVIDIA's weights, on a mix of LibriSpeech audiobooks (50%), synthetic speech of modern
text (30%) and 525 field recordings (20%), with recorded background noise mixed in; then
converted like the originals (`CITRINET_CKPT`, [`docs/MODELS.md`](../docs/MODELS.md#2-your-own-fine-tuned-variant)).
The fine-tuning cost some clean-speech accuracy; a training mix that keeps it is future work
([`docs/ROADMAP.md`](../docs/ROADMAP.md)).

Your own fine-tuned variants, other languages, other voices: [`docs/MODELS.md`](../docs/MODELS.md).

## License and attribution

These files are derived from **"stt_en_citrinet_256_ls" by NVIDIA**, licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Changes: converted to the mmrt
format, weights and activations quantized to 8 bits, and for the int4 files weights
re-quantized to 4-bit per-channel codebooks (GPTQ); for the `noisy` files, the weights were
also fine-tuned as described above. They are not covered by this repository's MIT license,
and redistributing them requires the same attribution.
