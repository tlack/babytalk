# Speech models

Ready-to-flash speech-to-text models for the runtime in `mmrt/`. Both are NVIDIA's
[Citrinet-256](https://huggingface.co/nvidia/stt_en_citrinet_256_ls) (English, trained on
LibriSpeech), converted and quantized by this project's `export/` scripts.

| file | weights | size | word error rate* | fits |
|---|---|---|---|---|
| `citrinet256_int4.mmrt` | 4-bit (first and last layers 8-bit) | 5.99 MB | 8.2% | the standard 6 MB `model` partition |
| `citrinet256_int8.mmrt` | 8-bit | 9.78 MB | 6.3% | the int8 layout's 10 MB partition |

\*LibriSpeech test-clean, 655 utterances, lower is better. The original float model: 3.8%.

SHA-256:

```
bdb6be064984b86af7c7ead182b2852420edaae252733264cd76f81bad0f8d47  citrinet256_int4.mmrt
6fd16badafdb2d02358682faec3055de93e011958d6b1431222ef27c66b8b493  citrinet256_int8.mmrt
```

## License and attribution

These files are derived from **"stt_en_citrinet_256_ls" by NVIDIA**, licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Changes: converted to the mmrt
format, weights and activations quantized to 8 bits, and for the int4 file weights
re-quantized to 4-bit per-channel codebooks (GPTQ). They are not covered by this
repository's MIT license, and redistributing them requires the same attribution.
