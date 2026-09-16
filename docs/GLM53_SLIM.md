# GLM 5.3 SLIM (expert-pruned full GLM 5.3)

[README](../README.md) | [Models](MODELS.md) | [GGUF tools](../gguf-tools/README.md)

This fork adds support for expert-pruned derivatives of the full GLM 5.3
checkpoint, in particular **GLM-5.3-SLIM-E192**: every MoE layer keeps 192 of
the original 256 routed experts, no MTP block is shipped, and everything else
(MLA + DSA attention, shared experts, dense layers, router, tokenizer, chat
template) is identical to `zai-org/GLM-5.3`.

Upstream DwarfStar pins the GLM DSA shape to the official checkpoint (79
blocks, 256 routed experts, one MTP block).  The fork derives the routed expert
count, the block count and the MTP block count from the GGUF metadata instead,
both in the quantizer and in the runtime, and keeps every other dimension
strictly validated.

## Why

The published full GLM 5.3 Q2 GGUF is 197 GiB.  With 192 experts the same
recipe (IQ2_XXS routed experts, Q8_0 everything else) is **149.7 GiB**, which
fits:

| Machine | Fit |
| --- | --- |
| 256 GB / 512 GB Mac Studio | resident, large context |
| 192 GB Mac Studio | resident with a modest context, or SSD streaming |
| 128 GB Mac / DGX Spark | [SSD streaming](SSD_STREAMING.md) only |
| one 180 GB B200 | resident, about 29 GiB left for context |

## Build

```sh
make                # Metal
make cuda-generic   # CUDA, local GPU architecture
```

Nothing else changes: `ds4`, `ds4-server`, `ds4-agent` and `ds4-eval` accept
the SLIM GGUF like any other GLM GGUF.  Startup prints
`ds4: GLM DSA variant: 192 routed experts, 78 blocks, 0 MTP block(s)`.

## Quantize

The source is the FP8 (block 128) Hugging Face snapshot.  A GLM GGUF with the
same tokenizer is needed as tokenizer template; the published full GLM 5.3 Q2
file works and only its metadata is read.

```sh
make -C gguf-tools
python3 gguf-tools/glm53_full_quantize.py \
  --hf /path/to/GLM-5.3-SLIM-E192 \
  --tokenizer-template gguf/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --model-name GLM-5.3-SLIM-E192 \
  --repo-url https://huggingface.co/cloudyu/GLM-5.3-SLIM-E192 \
  --cuda --cuda-batch 32 \
  --out gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf
```

Drop `--cuda` (and add `--threads N`) for a CPU-only conversion.

Use `--dry-run` first: it validates the checkpoint index, the FP8 scales and
the tokenizer, and prints the exact output size without reading tensor data.
For SLIM-E192 the plan is 1782 tensors / 160,760,301,792 bytes:

```text
type_bytes: IQ2_XXS 140142182400   (routed gate/up/down, 75 layers x 192)
type_bytes: Q8_0     20188545024   (attention, shared experts, dense FFN, embedding, output)
type_bytes: F32        420030720   (norms, routers, indexer projections)
```

On the CPU, IQ2_XXS costs about 7 s per 2048x6144 expert matrix per core and
there are 43,200 of them (nine hours on 20 cores).  With a CUDA GPU add
`--cuda --cuda-batch 32`: the routed experts are quantized by the PyTorch port
in `gguf-tools/iq2xxs_cuda.py`, byte-identical to the C quantizer for the
same importance vector, at about 30 ms per expert on a B200 (the whole model
in about 30 minutes, ~25 GiB of VRAM).  See the
[GGUF tools README](../gguf-tools/README.md#iq2_xxs-on-a-cuda-gpu).
`--resume` continues an interrupted run.  `--provisional-q2k` swaps the routed
experts to Q2_K (185 GiB) when a calibration model is wanted.

### Imatrix

Without `--imatrix`, IQ2_XXS uses the weight-energy fallback described in the
[GGUF tools README](../gguf-tools/README.md#when-no-imatrix-is-given).  The
imatrix of the original 256-expert model cannot be reused directly because the
kept experts are renumbered.  To collect one for the pruned model, run the
fallback GGUF through the runtime collector and quantize again:

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/GLM-5.3-SLIM-E192-routed-imatrix.dat --ctx 32768
python3 gguf-tools/glm53_full_quantize.py ... \
  --imatrix gguf/GLM-5.3-SLIM-E192-routed-imatrix.dat \
  --out gguf/GLM-5.3-SLIM-E192-IQ2_XXS-imatrix.gguf
```

## Run

```sh
# Metal, resident
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 32768

# Metal, 128 GB machine
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ssd-streaming --ctx 16384

# CUDA, one GPU
./ds4-server -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --gpu-devices 0 --gpu-vram auto --ctx 65536
```

Notes:

* There is no MTP block: do not pass `--mtp`.
* SSD streaming starts cold for pruned variants.  The built-in GLM 5.2 expert
  hot seed uses the original expert numbering and is skipped; the cache
  demand-fills.  A dedicated hot list can be produced later with
  `--expert-profile`.
* GLM requires `--power 100`, as upstream.
* The SLIM base model itself trades Chinese knowledge QA (C-Eval -7 pt) for
  size; coding, tool calling and math are within noise of GLM 5.3 according to
  its model card.  The 2-bit routed experts add their own loss on top of that;
  measure with `ds4-eval` before relying on a recipe.

## Evaluate

```sh
./ds4-eval -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --suite core --plain --trace /tmp/slim-core.txt
./ds4-eval -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --suite hard --retry-incomplete --plain
```

Add `--cuda --gpu-devices 0 --gpu-vram auto` on CUDA hosts.  Compare against
the published full GLM 5.3 Q2 on the same suite and machine; absolute numbers
depend on the generation budget.

## What changed in this fork

* `gguf-tools/glm53_manifest.py`: `glm53_full_spec(config)` derives block
  count, MTP presence, routed expert count and indexer owner layers from
  `config.json`; the index validator takes that spec.
* `gguf-tools/iq2xxs_cuda.py`: IQ2_XXS routed-expert quantization on CUDA,
  byte-identical to `quants.c`; `--cuda` in both GLM quantizers.
* `gguf-tools/glm53_full_quantize.py`: no hardcoded 256/79/78; GGUF metadata
  (`glm-dsa.expert_count`, `glm-dsa.block_count`,
  `glm-dsa.nextn_predict_layers`) follows the checkpoint; `--model-name` and
  `--repo-url`.
* `ds4.c`: `config_validate_glm_dsa_model` accepts the GGUF's expert count,
  block count and MTP block count (bounded by `DS4_MAX_EXPERT`,
  `DS4_MAX_LAYER`); the GLM 5.2 streaming hot seed is only applied to the
  original shape.

The official GLM 5.2 / 5.3 GGUFs keep working unchanged.
