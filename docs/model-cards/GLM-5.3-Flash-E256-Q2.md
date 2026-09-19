---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- zh
tags:
- gguf
- glm5_next
- moe
- expert-pruned
- iq2_xxs
- dwarfstar
- ds4
- dgx-spark
- gb10
---

# GLM-5.3-Flash-E256 Q2 — GLM 5.3 Flash for the DGX Spark

**A 78.9 GiB GGUF of GLM-5.3-Flash built for the 128 GB DGX Spark (GB10): 11 GiB smaller than the
stock DwarfStar Q2, same speed, ~40 GiB of unified memory left for context and the rest of the
system.** Runs with [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim), a fork of
[DwarfStar](https://github.com/antirez/ds4).

GLM 5.3 Flash with 256 of its 288 routed experts per layer (the 32 least-used removed), routed experts
in 2-bit (IQ2_XXS / Q2_K), attention, dense layers and shared experts in 8-bit — the same tensor
layout DwarfStar's Spark CUDA kernels are tuned for. Vision works with DwarfStar's stock GLM 5.3 Flash
encoder file (tested); no MTP draft block.

| | Stock DwarfStar GLM 5.3 Flash Q2 | **GLM-5.3-Flash-E256 Q2** |
|---|---|---|
| Size | 90 GiB | **78.9 GiB** |
| Routed experts / layer (active per token) | 288 (8) | **256 (8)** |
| Headroom on a 128 GB Spark after weights + graph | ~25 GiB | **~40 GiB** |
| Comfortable context on a Spark | 16–32 K | **64 K+** (KV ≈ 0.05 GiB per 4 K tokens) |
| Generation speed | same | same (identical work per token) |
| Vision (`--vision` + stock encoder GGUF) | yes | **yes** (tested) |
| MTP speculative decoding | yes | no |
| Runtime | `antirez/ds4` | `yuhai-china/ds4-glm-slim` (upstream pins 288 experts) |

## File

| File | Bytes | SHA-256 |
|---|---:|---|
| `glm-5.3-flash-e256-q2.gguf` | 84,694,295,648 | see `glm-5.3-flash-e256-q2.gguf.sha256` |

## Quick start on a DGX Spark

```sh
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4 && cd ds4
make cuda-spark                                              # GB10 build
mkdir -p gguf && mv /path/to/glm-5.3-flash-e256-q2.gguf gguf/

./ds4        -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536     # chat
./ds4-server -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536     # OpenAI-compatible API on :8000
./ds4-agent  -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536     # built-in coding agent
```

Images: add DwarfStar's stock GLM 5.3 Flash encoder (1.1 GB, `./download_model.sh glm53-vision`) — the
pruning did not touch the vision tower or the projector, so the unmodified encoder matches this file:

```sh
./ds4-server -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536 \
  --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf      # PNG/JPEG via OpenAI/Anthropic image blocks
./ds4 … --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf  # then /read image.png in the chat
```

Startup prints `ds4: GLM 5.3 Flash variant: 256 routed experts, 45 blocks, 0 MTP block(s)`. The
model stays in unified memory (no copy); first load is limited by reading 79 GiB from disk, so keep
the file on the internal NVMe. Thinking is on by default — `--nothink` for direct answers, `/think`
and `/nothink` in the chat, and the server honours the GLM `reasoning_effort` field
(`low`/`high`/`max`). Do not pass `--mtp`. For several concurrent users:
`./ds4-server … --ctx 32768 --batched-session 4`. Client setup (Pi, OpenCode, Codex CLI, Claude Code)
is in the fork's `docs/CLIENTS.md`.

Also runs on a 128 GB Apple Silicon Mac (`make`, drop `--cuda`) and on discrete NVIDIA GPUs with
≥ 90 GB of VRAM (`make cuda-generic`).

## Performance

Measured on one NVIDIA B200 (the fork's discrete-GPU path, model resident in VRAM, single stream):

| | |
|---|---:|
| Generation, short context | 38 t/s |
| Generation, sustained 16 K-token reasoning | 30 t/s |
| Prefill | 84 t/s on a 60-token prompt; grows with prompt length |

Expected on a DGX Spark: identical to DwarfStar's recorded stock GLM 5.3 Flash Q2 baseline, because
the active weights read per token are the same (8 of the remaining experts + the same 8-bit
attention/shared layers). From `docs/PERFORMANCE.md` of DwarfStar:

| Context | Prefill | Generation |
|---:|---:|---:|
| 2 K | 826 t/s | 18.1 t/s |
| 16 K | 872 t/s | 15.1 t/s |
| 32 K | 856 t/s | 14.4 t/s |
| 64 K | 823 t/s | 13.8 t/s |

Spark decode is bound by LPDDR5X bandwidth; pruning experts saves memory, not bandwidth, so the gain
is the extra headroom, not speed. The author validated this file on the B200 only; the Spark and Mac
code paths are upstream's, exercised daily by the stock Q2.

## Quality

**Expert pruning cost (256 vs 288 experts, measured at 4-bit under vLLM, one B200):**

| Benchmark | 256-expert GLM 5.3 Flash, 4-bit |
|---|---:|
| HumanEval | 97.6 |
| GPQA-Diamond (low effort) | 77.3 |
| C-Eval | 89.4 |
| AIME 2025 | 74.2 |
| MMMU | 76.1 |
| BFCL Non-Live / Live | 87.7 / 80.5 |
| BFCL multi-turn base / miss-func | 73.0 / 75.0 |

**This 2-bit file (`ds4-eval`, DwarfStar's built-in harness; thinking on, 16 000-token budget,
greedy; first 40 core cases, one B200):**

| Set | Passed |
|---|---:|
| GPQA Diamond | 11 / 14 |
| SuperGPQA | 12 / 13 |
| AIME 2025 | 12 / 13 |
| **Total** | **35 / 40** (3 wrong, 2 out of budget) |

An imatrix-guided build of the same layout was tested head-to-head and did worse (31 / 40, with
twice the reasoning tokens and five budget exhaustions), so this weight-energy-importance build is
the one published.

Qualitative checks: correct answers to bilingual common-sense and medical questions, no
Chinese/English mixing, correct code generation (e.g. a palindrome checker with Chinese test strings).

**Vision** (this file + stock encoder, `ds4-server`, B200): shapes/colours/text in a synthetic image
described correctly in Chinese ("红色的圆形 … 蓝色的正方形 … DGX SPARK 128GB"); a bar chart read
correctly (both values, the 11.1 GiB / 12.3 % difference computed). ~6 s per image request including
encoding (377–385 prompt tokens). The 256-expert model scores MMMU 76.1 at 4-bit under vLLM.

## What is in the file

| Role | Type | Bytes |
|---|---|---:|
| Routed experts gate / up (42 MoE layers × 256) | IQ2_XXS | 46.5 GB |
| Routed experts down | Q2_K | 29.6 GB |
| KDA linear attention, DSA attention, dense FFN, shared experts | Q8_0 (1.3 GB Q4_K) | 8.2 GB |
| Embedding, output head | BF16 / Q8_0 | 1.3 GB |
| Norms, routers, hyper-connections, indexer | F32 | 0.2 GB |

Architecture otherwise unchanged: 45 layers (3 dense + 42 MoE), KDA linear attention with DSA every
fourth layer, hyper-connections, top-8 routing + 1 shared expert, 154 880-token vocabulary, GLM 5.3
chat template and tool calling.

## Limitations

* 2-bit routed experts on top of an 11 % expert pruning: chat, coding and agent use are the target;
  expect a drop on the hardest maths/science reasoning versus the 4-bit vLLM deployment.
* Requires the fork; upstream DwarfStar rejects the file (`expected expert_count=288`).
* No MTP head (no speculative decoding). The vision encoder is a separate file (upstream's), not bundled here.
* Not run on a DGX Spark by the author.

License: MIT (base model). Credits: Z.AI (GLM-5.3-Flash); Salvatore Sanfilippo and the DwarfStar
contributors (runtime, Spark kernels, Q2 recipe); llama.cpp/GGML (quant formats);
[MOE-SLIM](https://github.com/yuhai-china/MOE-SLIM) (expert pruning); yuhai-china (this build).
