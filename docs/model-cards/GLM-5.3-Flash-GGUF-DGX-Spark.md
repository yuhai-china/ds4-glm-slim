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
- llama.cpp
- dgx-spark
- gb10
- glm5-next
- moe
- expert-pruned
- iq2_xxs
---

# GLM-5.3-Flash-GGUF-DGX-Spark — GLM 5.3 Flash for the DGX Spark, with llama.cpp

**The GLM 5.3 Flash you can keep resident on a 128 GB DGX Spark with room to spare: 79.1 GiB
instead of the stock 90 GiB Q2, 32 of 288 routed experts removed per layer, everything else
untouched. Runs in llama.cpp (`glm5-next`), so you get the OpenAI-compatible `llama-server`,
continuous batching, tool calling and `reasoning_content` on the Spark.**

| | Stock GLM-5.3-Flash Q2 GGUF | **This file (`GLM-5.3-Flash-Q2-DGX-Spark.gguf`)** |
|---|---|---|
| Size | 90 GiB | **79.1 GiB** |
| Routed experts per layer (active per token) | 288 (8) | **256 (8)** |
| Free unified memory on a 128 GB Spark after weights | ~25 GiB | **~40 GiB** |
| Context that fits comfortably on a Spark | 16–32 K | **64 K, or 4 × 16 K sessions** |
| Decode speed | same class (identical work per token) | same class |
| Quality (see below) | reference | 4-bit: HumanEval 97.6, C-Eval 89.4, GPQA-D 77.3; 2-bit harness A/B on par or better |

The expert selection was calibrated on a bilingual code / agent / science / maths mix; attention
(KDA + DSA), dense layers, shared experts, router, tokenizer and chat template are unchanged.

## Download

```sh
hf download autotrust/GLM-5.3-Flash-GGUF-DGX-Spark --local-dir ./GLM-5.3-Flash-GGUF-DGX-Spark
```

(private repository — `hf auth login` with an account in the `autotrust` org first). Files:
`GLM-5.3-Flash-Q2-DGX-Spark.gguf` (79.1 GiB) and its `.sha256`.

## DGX Spark quick start

GLM-5.3-Flash support is in llama.cpp pull request
[#27773](https://github.com/ggml-org/llama.cpp/pull/27773) (`glm5-next`; not merged at the time of
writing — once it is, plain `master` works).

```sh
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git fetch origin pull/27773/head:glm5next && git checkout glm5next
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real     # GB10
cmake --build build --config Release -j

# OpenAI-compatible API on :8080, 4 sessions x 16K, continuous batching
./build/bin/llama-server -m GLM-5.3-Flash-Q2-DGX-Spark.gguf -ngl 99 -fa on \
    -c 65536 -np 4 --cont-batching --host 0.0.0.0 --port 8080

# single long-context chat
./build/bin/llama-cli -m GLM-5.3-Flash-Q2-DGX-Spark.gguf -ngl 99 -fa on -c 65536
```

* Keep the file on the internal NVMe; the first load reads 79 GiB. Stop other GPU work first.
* Memory on the Spark: 79.1 GiB weights + ~1 GiB per 16 K tokens of context (KDA layers keep a
  constant state; the DSA layers use a compact cache) + a few GiB of compute buffers. `-c 65536`
  leaves ~30 GiB free; `-np 4 -c 65536` (4 × 16 K) is a good multi-user setting.
* Thinking is on by default (the GLM template opens `<think>`). `--reasoning-budget 0` disables it,
  `--reasoning-budget 4096` caps it, `--chat-template-kwargs '{"reasoning_effort":"low"}'` selects
  the template's low / high / max effort. The server returns thinking separately as
  `reasoning_content`; tool calls come back as OpenAI `tool_calls`.
* Expected decode speed on a Spark: the same class as the stock Flash Q2 — the per-token work is
  identical (8 experts + the same 8-bit attention). Decode is bound by the 273 GB/s LPDDR5X (about
  11 GB of weights per token), so roughly 15–20 t/s single-stream; batching several sessions gives
  more aggregate throughput. Not measured by the author on a Spark; see the B200 table below for
  relative numbers.
* Two Sparks (ConnectX link) are not needed for this file; it is a single-Spark model.

Also runs on: 128 GB Apple Silicon (`cmake -B build` without CUDA), discrete NVIDIA GPUs with
≥ 90 GB, or partially offloaded (`-ngl N`) on smaller cards.

## Speed reference (llama.cpp, CUDA, one B200, model resident)

| | tokens/s |
|---|---:|
| Prompt processing (pp512 / pp2048) | 1037 / 1086 |
| Generation, 1 sequence | 54 |
| Generation, 4 / 8 / 16 / 32 parallel sequences (aggregate) | 130 / 170 / 207 / 253 |

(`llama-bench`, `llama-batched-bench`; 256-token prompts, 128 generated tokens, flash attention.)

## Quality

The 4-bit model with the same 256-expert selection, under vLLM (one B200): HumanEval 97.6,
C-Eval 89.4, GPQA-Diamond 77.3 (low effort), AIME 2025 74.2, MMMU 76.1, BFCL Non-Live 87.7 /
Live 80.5 / multi-turn 73–75 — the pruning itself costs little.

This 2-bit file, on DwarfStar's `ds4-eval` harness (GPQA Diamond, SuperGPQA, AIME 2025 interleaved;
thinking on, 16 000-token budget, greedy), first 40 core cases, compared with the 2-bit 744B
GLM-5.3-SLIM-E192 on the same cases:

| 2-bit GGUF | Passed | Wrong | Out of budget |
|---|---:|---:|---:|
| **GLM-5.3-Flash-Q2-DGX-Spark (this file, 79 GiB; 256 experts)** | **35 / 40** | 3 | 2 |
| GLM-5.3-Q2-DGX-Spark (the 744B build, 150 GiB; [autotrust/GLM-5.3-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark)) | 30 / 40 | 1 | 9 |

An imatrix-guided build of the same layout scored 31 / 40 with twice the reasoning tokens, so this
weight-energy-importance build is the one published. Qualitative checks: correct bilingual
common-sense, medical and coding answers; no Chinese–English mixing.

## What is in the file

| Role | Type | Bytes |
|---|---|---:|
| Routed experts gate / up (42 MoE layers × 256) | IQ2_XXS | 46.5 GB |
| Routed experts down | Q2_K | 29.6 GB |
| KDA linear attention, DSA attention, dense FFN, shared experts | Q8_0 (1.3 GB Q4_K) | 8.2 GB |
| Embedding, output head | Q8_0 | 1.3 GB |
| Norms, routers, hyper-connections, indexer, k-pool compressor | F32 | 0.5 GB |

45 layers (3 dense + 42 MoE), KDA linear attention with DSA every fourth layer (k-pool indexer),
hyper-connections, top-8 of 256 routed experts + 1 shared, 154 880-token vocabulary, GLM 5.3 chat
template with tool calling. Plain llama.cpp GGUF (architecture `glm5-next`). No MTP block; text
model (no vision projector).

## File

| File | Bytes | SHA-256 |
|---|---:|---|
| `GLM-5.3-Flash-Q2-DGX-Spark.gguf` | 84,929,449,952 | `44ef6ae232e4928803e6b485e165ec3731bebb49fd62ec3e2f4200730295d83a` |

## Limitations

* 2-bit routed experts on top of an 11 % expert pruning: chat, coding and agent use are the target;
  expect a drop on the hardest maths/science reasoning versus the 4-bit deployment.
* Needs the llama.cpp `glm5-next` branch until it is merged. Two quirks of that branch are patched in
  `llamacpp-pr27773-glm5next-lenient.patch` (in the tooling repository): tolerate u64 metadata, and
  return a reply verbatim instead of HTTP 500 when `max_tokens` cuts it mid-UTF-8-character. Neither
  is required to run the model.
* No MTP head, no vision projector in this file.

License: MIT (base model). Credits: Z.AI (GLM-5.3-Flash); llama.cpp / ggml and the `glm5-next` PR
authors; DwarfStar (2-bit recipe and quantizer); [MOE-SLIM](https://github.com/yuhai-china/MOE-SLIM)
(expert pruning); tooling: [ds4-glm-slim](https://github.com/yuhai-china/ds4-glm-slim); yuhai-china
(this build).
