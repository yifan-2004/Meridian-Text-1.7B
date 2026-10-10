# Scientific architecture figure

[Editable SVG](../assets/model-overview.svg) · [PNG](../assets/model-overview.png)

The four panels expand the actual checkpoint architecture and the local training objective. They do not introduce a new architecture or report an experiment.

- **Decoder:** 28 pre-norm layers, residual addition around attention and SwiGLU, hidden width 2048, and tied embedding/LM-head weights.
- **Attention:** 16 query heads and 8 key/value heads, each 128 channels. Q and K receive per-head RMS normalization before RoPE. One representative group illustrates how two query heads share one key/value head. The lower-triangular matrix illustrates causal masking; its colors are schematic, not measured attention.
- **SwiGLU:** two 2048→6144 projections; SiLU on the gate branch; elementwise multiplication; a 6144→2048 down projection.
- **Adaptation:** per-document tokenization with EOS, right padding, effective shifted next-token labels, masked padding, full-parameter FP32 AdamW and token-weighted validation. This matches `scripts/train.py`; it is separate from the upstream pretraining recipe.

Tensor grids show structure, not individual measured activations. The configured vocabulary is 151936. The 4096 label denotes this distribution's documented training context, not its larger configured positional limit.

## Sources

[Published configuration at source revision `39a3716`](https://huggingface.co/qvac/qvac-genesis-iii-qwen3-1.7b-combined/blob/39a371617fabe508eb793c9b7832ec3496d4e96e/config.json) provides all dimensions, RoPE theta and tied embeddings. [Transformers Qwen3 implementation, v4.53.3](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/qwen3/modeling_qwen3.py) provides the ordering of normalization, rotary positions, KV repetition and SwiGLU operations. The checkpoint is unchanged; see [provenance](PROVENANCE.md).

## Regenerate

```bash
python scripts/draw_figure.py
# macOS PNG export; SVG remains the editable source.
sips -s format png assets/model-overview.svg --out assets/model-overview.png
```

The original vector renderer uses only the Python standard library. `assets/model-layout.json` records the verified architecture metadata and source URLs; the renderer checks these dimensions before exporting. Typography uses common serif fonts with fallbacks. Layout: 2200 × 1470 px, white background, restrained semantic colors, explicit tensor dimensions and mathematical operators.

Original figure artwork: Yifan Ouyang, 2026; distributed under this repository's Apache-2.0 license. Architecture attribution remains with its upstream authors.
