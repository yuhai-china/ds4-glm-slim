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
- llama.cpp
- moe
- expert-pruning
- iq2_xxs
- glm
- glm-dsa
---

# GLM-5.3-SLIM-E192 — IQ2_XXS GGUF: GLM 5.3 on a single machine, with llama.cpp

**GLM 5.3, the 744B-parameter frontier MoE, running on one GPU or one Mac.**
Expert pruning (192 of 256 routed experts, "SLIM") took 25 % off the model;
2-bit routed experts took the rest. The result is a **149.7 GiB** file that
runs in **llama.cpp** (architecture `glm-dsa`, upstream) — 42 t/s single-stream
and 177 t/s aggregate with 32 parallel requests on one 180 GB B200, or a single
Mac Studio (256 GB; 192 GB with a small context) — hardware where the unpruned
model needs a multi-GPU node or does not fit at all.

| GLM 5.3 form | Size | What it takes to run it |
|---|---:|---|
| Original FP8 (`zai-org/GLM-5.3`) | 756 GB | 8× H200/B200 or 4× B300, tensor parallel (88 GiB/GPU at TP=8: too big for 80 GB cards) |
| Pruned FP8 (`cloudyu/GLM-5.3-SLIM-E192`) | 564 GB | 4× B200/B300 or 8× 80 GB cards |
| Full GLM 5.3 IQ2_XXS GGUF (`antirez/glm-5.3-gguf`) | 197 GiB | 256 GB+ Mac resident; 128 GB Mac via SSD streaming; **does not fit one 180 GB GPU** |
| **This file — SLIM IQ2_XXS GGUF** | **149.7 GiB** | **one 180 GB GPU resident (llama.cpp: 42 t/s single, 177 t/s batched); one Mac Studio resident (256 GB comfortably, 192 GB with a small context)** |

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

It is a standard llama.cpp GGUF (`glm-dsa` architecture, the same tensor names,
MLA `attn_k_b`/`attn_v_b` split and metadata as llama.cpp's own GLM 5.2/5.3
conversions); the recommended runtime is **llama.cpp** (see below). The
DwarfStar fork [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim)
also loads it.

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
| Runtime | **llama.cpp** (upstream, CUDA / Metal / CPU); DwarfStar fork `ds4-glm-slim` also works |
| Fits | **one** 180 GB GPU (B200/GB200) resident; **one** Mac Studio 256 GB resident, 192 GB resident with a small context. Not a 128 GB machine target (see GLM-5.3-SLIM-E160 / Flash-E256 for those) |
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

## Run it with llama.cpp

`glm-dsa` (GLM 5.2 / 5.3 with the DSA indexer) is supported by upstream llama.cpp; nothing
special is needed:

```sh
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j      # NVIDIA
# Apple Silicon: cmake -B build && cmake --build build --config Release -j

# OpenAI-compatible server, 8 slots x 16K context, continuous batching
./build/bin/llama-server -m GLM-5.3-SLIM-E192-IQ2_XXS.gguf -ngl 99 -c 131072 -np 8 --cont-batching -fa on
# interactive
./build/bin/llama-cli -m GLM-5.3-SLIM-E192-IQ2_XXS.gguf -ngl 99 -c 32768
```

Thinking is on by default (the template opens `<think>`); `--reasoning-budget 0` disables it,
`--chat-template-kwargs '{"reasoning_effort":"low"}'` selects the effort level, and the server
returns reasoning separately as `reasoning_content`. Tool calling works through the GLM template.

Memory on a 180 GB card: 149.7 GiB weights + ~1.2 GiB per 16 K tokens of context (compressed
MLA/DSA cache); `-np 8 -c 131072` fits. On a 192 GB Mac keep the context small; on 256 GB it is
comfortable.

One optional patch (`docs/llamacpp/llamacpp-master-glm-dsa-lenient.patch` in the fork) makes
`llama-server` return a reply verbatim instead of HTTP 500 when `max_tokens` cuts it in the middle
of a multi-byte character; it is not needed to run the model.

## Quality

The pruned FP8 base ([model card](https://huggingface.co/cloudyu/GLM-5.3-SLIM-E192))
was measured against GLM-5.3 by its author: coding, cybersecurity, tool
calling and math within run-to-run noise; GPQA-Diamond −3.6 pt; C-Eval −7.3 pt
(the deliberate trade-off). The 2-bit routed experts of this file add their
own loss on top. Held-out perplexity of the FP8 base vs. the original: code
+1.0 %, English chat +4.7 %, Chinese +7.3 %.

This GGUF, DwarfStar's `ds4-eval` harness on the same weights (GPQA Diamond,
SuperGPQA, AIME 2025, COMPSEC; thinking on, default budgets), one B200:

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

A per-layer non-uniform pruning study on this checkpoint (mass-greedy and
layer-sensitivity allocations at the same budget) found no gain over uniform
expert counts; the uniform 160-expert sibling GLM-5.3-SLIM-E160 (128 GiB,
held-out PPL +3.6 %) is the smaller option.

## Speed

llama.cpp, CUDA, one NVIDIA B200 (180 GB), model resident, flash attention on
(`llama-bench` / `llama-batched-bench`, 256-token prompts, 128 generated tokens):

| | tokens/s |
|---|---:|
| Prompt processing (pp512 / pp2048) | 730–760 |
| Generation, 1 sequence | 42 |
| Generation, 4 / 8 / 16 / 32 parallel sequences (aggregate) | 96 / 120 / 154 / 177 |

Per-token work is the same as the full GLM 5.3 Q2 (8 routed experts per layer either way), so the
gain over the 197 GiB file is memory, not speed. Metal figures for the full GLM 5.3 Q2 on the same
class of Mac apply directly.

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
* **No MTP head**, no imatrix (weight-energy importance; an imatrix A/B on the
  Flash sibling did not favour imatrix at this bit width).
* **192 GB Macs** are borderline: 149.7 GiB weights + graph + context leaves
  little for macOS; keep the context small.
* **128 GB machines (DGX Spark, 128 GB Mac)** cannot hold the file; use
  GLM-5.3-Flash-E256 Q2 (78.9 GiB) or the smaller SLIM builds instead.
* `general.source.revision` in the GGUF metadata carries the quantizer's
  default (the official GLM-5.3 revision); the SLIM checkpoint revision used
  is `e45b62eb3f5a22232f1e4980da255266ab933f31`.

## Verify the download

```text
size    160760301792 bytes
sha256  0b40a1739e674a2a850000851d771ec4cb4f7662cf6152b52ef1d0984c90cf72   (header: context_length stored as u32 for llama.cpp; weights unchanged)
```

## License and credits

* Weights: GLM-5.3 License (Z.AI), including the Model-as-a-Service clause;
  this file is a derivative of `zai-org/GLM-5.3` via `cloudyu/GLM-5.3-SLIM-E192`.
* Expert pruning: cloudyu (GLM-5.3-SLIM-E192).
* Inference engine and quant recipe: Salvatore Sanfilippo and the DwarfStar
  contributors (`antirez/ds4`), building on llama.cpp / GGML.
* Fork, CUDA quantizer and this build: yuhai-china — https://github.com/yuhai-china/ds4-glm-slim.
