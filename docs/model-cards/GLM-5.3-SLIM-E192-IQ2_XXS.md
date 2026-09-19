---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model:
- cloudyu/GLM-5.3-SLIM-E192
- zai-org/GLM-5.3
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- moe
- expert-pruning
- iq2_xxs
- glm
- dwarfstar
- ds4
---

# GLM-5.3-SLIM-E192 — IQ2_XXS GGUF: GLM 5.3 on a single machine

**GLM 5.3, the 744B-parameter frontier MoE, running on one GPU or one Mac.**
Expert pruning (192 of 256 routed experts, "SLIM") took 25 % off the model;
DwarfStar's 2-bit routed-expert recipe took the rest. The result is a
**149.7 GiB** file that stays fully resident on a single 180 GB GPU (B200,
21 tokens/s measured) or a single Mac Studio (256 GB; 192 GB with a small
context), and runs on a **128 GB DGX Spark** or 128 GB Mac through DwarfStar's
SSD streaming (bounded expert cache in memory, the rest read from NVMe) —
hardware where the unpruned model needs a multi-GPU node or does not fit at all.

| GLM 5.3 form | Size | What it takes to run it |
|---|---:|---|
| Original FP8 (`zai-org/GLM-5.3`) | 756 GB | 8× H200/B200 or 4× B300, tensor parallel (88 GiB/GPU at TP=8: too big for 80 GB cards) |
| Pruned FP8 (`cloudyu/GLM-5.3-SLIM-E192`) | 564 GB | 4× B200/B300 or 8× 80 GB cards |
| Full GLM 5.3 IQ2_XXS GGUF (`antirez/glm-5.3-gguf`) | 197 GiB | 256 GB+ Mac resident; 128 GB Mac via SSD streaming; **does not fit one 180 GB GPU** |
| **This file — SLIM IQ2_XXS GGUF** | **149.7 GiB** | **one 180 GB GPU resident (21 t/s); one Mac Studio resident (256 GB comfortably, 192 GB with a small context); one DGX Spark or 128 GB Mac via SSD streaming (CUDA streaming path verified with this file)** |

The pruning is the enabler: at 2 bits the unpruned experts alone are 187 GB,
so no single-device quantization of the original could fit a 180 GB card with
room for a context. Pruning removes 47 GB of expert bytes at this precision
and, per its author's A/B on the FP8 checkpoints, costs nothing measurable on
coding, cybersecurity, tool calling and math (GPQA −3.6 pt, C-Eval −7.3 pt).

What the file is: GLM-5.3-SLIM-E192 (attention with MLA + DSA sparse indexer,
shared experts, router, tokenizer and chat template identical to GLM 5.3; no
MTP head) with routed experts in IQ2_XXS (2.06 bits/weight) and everything
else in Q8_0 — the same recipe DwarfStar publishes for the full GLM 5.3, so
per-token compute and memory traffic are unchanged; only the footprint drops.

It runs with the **[`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim)
fork** of DwarfStar (a few changes on top of upstream `antirez/ds4`, see
[Requirements](#requirements)). It is not a llama.cpp GGUF: the tensor layout,
quant mix and metadata follow DwarfStar's GLM-DSA format.

## At a glance

| | |
|---|---|
| File | `GLM-5.3-SLIM-E192-IQ2_XXS.gguf` |
| Size | 160,760,301,792 bytes (149.7 GiB) |
| GGUF | v3, architecture `glm-dsa`, 1782 tensors |
| Parameters | ≈563 B total, ≈40 B active per token (8 of 192 routed experts + 1 shared) |
| Layers | 78 (3 dense + 75 MoE), no MTP block |
| Context | 1,048,576 positions in metadata; use what your memory allows |
| Routed experts | IQ2_XXS, 2.0625 bits/weight, weight-energy importance (no imatrix) |
| Everything else | Q8_0 (attention, shared experts, dense FFN, embeddings, output head); F32 norms/routers/indexer projections |
| Runtime | DwarfStar fork [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim) (Metal, CUDA; ROCm untested) |
| Fits | **one** 180 GB GPU (B200/GB200) resident; **one** Mac Studio 256 GB resident, 192 GB resident with a small context; **one DGX Spark (GB10, 128 GB)** or 128 GB Mac via SSD streaming |
| Source checkpoint | [`cloudyu/GLM-5.3-SLIM-E192`](https://huggingface.co/cloudyu/GLM-5.3-SLIM-E192) (FP8), revision `e45b62eb` |
| License | GLM-5.3 (same as the base model) |

## What is in the file

| Role | Tensors | Type | Bytes |
|---|---|---:|---:|
| Routed experts gate / up / down (75 layers × 192 experts) | 225 | IQ2_XXS | 140.14 GB |
| Attention (MLA q_a/q_b/kv_a/kv_b/o, DSA indexer q_b/k) | 78 layers | Q8_0 | 14.50 GB |
| Shared experts gate / up / down | 75 layers | Q8_0 | 3.01 GB |
| Dense FFN (layers 0–2) | 3 layers | Q8_0 | 0.72 GB |
| Token embedding, output head | 2 | Q8_0 | 2.02 GB |
| Norms, routers, bias, indexer weights_proj | — | F32 | 0.42 GB |

Per-token decode reads the same amount of data as the full GLM 5.3 Q2 (8
routed experts per layer either way), so **speed per token is the same as the
197 GiB file; the gain is memory**.

## Requirements

Upstream `antirez/ds4` cannot load this file: its GLM-DSA loader pins the
shape to the official checkpoint (256 experts, 79 blocks, 1 MTP block) and its
CUDA backend has no kernels for IQ2_XXS down projections. The fork adds:

* runtime acceptance of expert-pruned GLM 5.2/5.3 shapes (expert count, block
  count and MTP count read from the GGUF);
* CUDA kernels for all-IQ2_XXS routed layers (prefill on the mmq tier, decode
  on mmvq vector kernels) and resident weights on discrete GPUs;
* a CUDA quantizer for IQ2_XXS (byte-identical to the C one) and the tooling
  that produced this file.

Everything else — CLI, agent, HTTP server, KV snapshots, tool calling,
thinking control — is unchanged DwarfStar. Build it:

```sh
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4
cd ds4
make                # Apple Silicon / Metal
make cuda-generic   # NVIDIA, local GPU architecture (needs nvcc + cuBLAS)
```

Startup prints `ds4: GLM DSA variant: 192 routed experts, 78 blocks, 0 MTP block(s)`
when the fork recognises the file.

## Quick start

Put the file in `gguf/` inside the repository (or pass a full path with `-m`).

```sh
# interactive chat (thinking on by default; --nothink for direct answers)
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 32768

# OpenAI-compatible server on http://127.0.0.1:8000
./ds4-server -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 65536

# native coding agent
./ds4-agent -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 65536
```

Add `--cuda` on NVIDIA hosts. On a 128 GB Mac add `--ssd-streaming`. See
[RUNNING.md](RUNNING.md) for memory planning, streaming, server usage and
troubleshooting.

## DGX Spark (GB10, 128 GB)

The whole 744B-class GLM 5.3 on a desktop box. 149.7 GiB does not fit the
Spark's 128 GB, so DwarfStar's **SSD streaming** is used: attention, dense
layers, shared experts, embeddings and the router (19.2 GiB) stay resident, a
bounded cache holds the most-used routed experts, and the rest are read from
the internal NVMe on demand.

```sh
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4 && cd ds4
make cuda-spark                                  # GB10 build (sm_121)
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --ssd-streaming --ctx 16384
# explicit cache size (automatic budget otherwise); leave room for the OS:
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --ssd-streaming \
      --ssd-streaming-cache-experts 64GB --ctx 16384
./ds4-server -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --ssd-streaming --ctx 16384   # API on :8000
```

What to expect:

* Startup prints `ds4: GLM DSA variant: 192 routed experts, 78 blocks, 0 MTP block(s)`
  and a streaming plan such as `resident model 19.19 GiB + expert cache … + KV …`.
  With a 64 GiB cache about 7,000 of the 14,400 expert slots (each 9.28 MiB) are
  resident — half of the model's experts, versus under a third for the stock
  197 GiB full GLM 5.3 Q2 on the same machine, which is the point of the pruning
  for streaming: fewer, more useful experts to page.
* **Speed is NVMe-bound**: expect low single-digit tokens/s once the cache is
  warm (upstream measured 4–5 t/s for the 197 GiB full model with a 61 GiB cache
  on a 128 GB M5 Max; this file has 27 % fewer expert bytes to page). The first
  requests are slower while the cache fills — pruned variants start cold because
  DwarfStar's built-in GLM hot-expert seed uses the original expert numbering.
  Keep the file on the internal NVMe, not USB storage.
* Thinking and long prompts multiply the paging; `--nothink` and 16–32 K context
  are the practical settings. Vision: the model has no vision encoder.
* **If you want speed rather than the biggest model on a Spark**, use the
  resident 78.9 GiB GLM-5.3-Flash-E256 Q2 (~18 t/s on a Spark). This file is
  for when the answer quality of the 744B model matters more than latency.

The CUDA streaming path was exercised with this exact file on our B200 (cache
40 GiB, cold): correct Chinese output, plan as above. It has not been run on a
Spark by the author; the Spark build and streaming code are upstream's.

## Quality

The pruned FP8 base ([model card](https://huggingface.co/cloudyu/GLM-5.3-SLIM-E192))
was measured against GLM-5.3 by its author: coding, cybersecurity, tool
calling and math within run-to-run noise; GPQA-Diamond −3.6 pt; C-Eval −7.3 pt
(the deliberate trade-off). The 2-bit routed experts of this file add their
own loss on top. Held-out perplexity of the FP8 base vs. the original: code
+1.0 %, English chat +4.7 %, Chinese +7.3 %.

This GGUF, `ds4-eval` (DwarfStar's built-in harness: GPQA Diamond, SuperGPQA,
AIME 2025; thinking on, default budgets), CUDA B200:

| Set | Passed | Wrong | Out of budget |
|---|---:|---:|---:|
| GPQA Diamond (25) | 12 | 0 | 13 |
| SuperGPQA (25) | 17 | 4 | 4 |
| AIME 2025 (25) | 14 | 1 | 10 |
| COMPSEC cybersecurity (17) | 15 | 2 | 0 |
| **core total (92)** | **58** | 7 | 27 |

Runtime 14 h 28 min at ~11 t/s. Most misses are budget exhaustions on the
hardest GPQA/AIME items (the model keeps reasoning past the 16 000-token cap),
not wrong answers; only 3 of the 27 were repetition loops.

`ds4-eval` scores are integration checks, not leaderboard numbers; compare
against the published full GLM 5.3 Q2 run on the same machine and suite.

An imatrix-guided variant (importance collected from this model's own routed
activations with `ds4 --imatrix-dataset`) is the natural next step and would
replace this file's weight-energy importance.

## Speed

| Machine | Backend | Prefill | Decode |
|---|---|---:|---:|
| 1× NVIDIA B200 180 GB, model resident | CUDA | 44 t/s on a 30-token prompt (fixed cost dominated; mmq tier for long prompts) | 21 t/s short context, ~10–11 t/s during long (2–8 K) thinking |
| DGX Spark 128 GB, SSD streaming | CUDA | NVMe-bound | not measured by us; expect low single-digit t/s warm (see the Spark section) |
| Apple Silicon | Metal | not measured by us | not measured by us |

Metal figures for the full GLM 5.3 Q2 on the same class of Mac apply
directly (identical per-token work).

## How it was built

```sh
python3 gguf-tools/glm53_full_quantize.py \
  --hf GLM-5.3-SLIM-E192 \
  --tokenizer-template GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --model-name GLM-5.3-SLIM-E192 \
  --repo-url https://huggingface.co/cloudyu/GLM-5.3-SLIM-E192 \
  --cuda --cuda-batch 32 \
  --out GLM-5.3-SLIM-E192-IQ2_XXS.gguf
```

* Source: the FP8 (block 128) safetensors of `cloudyu/GLM-5.3-SLIM-E192`,
  dequantized exactly; tokenizer/chat template taken from the checkpoint,
  validated against the published full GLM 5.3 GGUF.
* Routed experts quantized on a B200 with the fork's PyTorch/CUDA port of
  DwarfStar's IQ2_XXS quantizer (`gguf-tools/iq2xxs_cuda.py`), which is
  byte-identical to the C implementation for the same importance vector
  (verified on 296,848 real blocks). 43,200 expert matrices in 29 minutes.
* Importance: DwarfStar's fallback `importance[column] = Σ row[column]²`
  (no activation imatrix).
* Attention, shared experts, dense FFN, embeddings and output head: Q8_0 via
  the C quantizer, as in the published GLM 5.3 Q2.

## Limitations

* **2-bit routed experts.** Expect a measurable drop versus the FP8 checkpoint
  on knowledge-heavy and Chinese-exam tasks; coding and agent use are the
  intended workloads.
* **No MTP head.** Do not pass `--mtp`.
* **No imatrix** in this build (see Quality).
* **192 GB Macs** are borderline: 149.7 GiB weights + ~5 GiB graph leaves
  little for macOS; DwarfStar's memory guard will ask for
  `DS4_GLM_MEMORY_GUARD_RESERVE_GB` to be lowered and the context kept small.
  Metal was not exercised by the author of this file; the code paths are
  upstream's, exercised by the 197 GiB full model.
* **SSD streaming starts cold** on pruned variants (the built-in GLM 5.2 hot
  seed uses the original expert numbering and is skipped); the first prompts
  on a DGX Spark or 128 GB Mac are slow while the expert cache fills.
* **DGX Spark is a streaming target, not a resident one**: low single-digit
  t/s, NVMe-bound. Not run on a Spark by the author.
* **Discrete CUDA** needs a single GPU with ≥ 158 GiB free (weights + 8 GiB)
  for the resident fast path; smaller cards need `--ssd-streaming` (or fall
  back to a host mapping that streams experts over PCIe at ~1 t/s). Multi-GPU
  placement is untested with this file.
* `general.source.revision` in the GGUF metadata carries the quantizer's
  default (the official GLM-5.3 revision); the SLIM checkpoint revision used
  is `e45b62eb3f5a22232f1e4980da255266ab933f31`.

## Verify the download

```text
size    160760301792 bytes
sha256  68abcb6effe4e7a4379d92f68607dfbd1aca0a1fad5dabda78a96f1a4a0e4725
```

## License and credits

* Weights: GLM-5.3 License (Z.AI), including the Model-as-a-Service clause;
  this file is a derivative of `zai-org/GLM-5.3` via `cloudyu/GLM-5.3-SLIM-E192`.
* Expert pruning: cloudyu (GLM-5.3-SLIM-E192).
* Inference engine and quant recipe: Salvatore Sanfilippo and the DwarfStar
  contributors (`antirez/ds4`), building on llama.cpp / GGML.
* Fork, CUDA quantizer and this build: yuhai-china — https://github.com/yuhai-china/ds4-glm-slim.
