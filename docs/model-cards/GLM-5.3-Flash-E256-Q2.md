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
- glm5-next
- moe
- expert-pruned
- iq2_xxs
- dgx-spark
---

# GLM-5.3-Flash-E256 Q2 — 78.9 GiB GGUF for llama.cpp

**GLM-5.3-Flash with 256 of its 288 routed experts, routed experts in 2-bit (IQ2_XXS gate/up,
Q2_K down), everything else 8-bit. One file, 79.1 GiB, runs in `llama.cpp` — 54 t/s single-stream
and 250 t/s with 32 parallel requests on one B200; fits a 128 GB DGX Spark or Mac with ~40 GiB to
spare for context.**

The 32 least-used experts per layer were removed after calibration on a bilingual code / agent /
science / maths mix (measured at 4-bit under vLLM: HumanEval 97.6, C-Eval 89.4, GPQA-Diamond 77.3,
AIME25 74.2, MMMU 76.1, BFCL Live 80.5). Attention (KDA + DSA), dense layers, shared experts,
router, tokenizer and chat template are unchanged.

## Run it with llama.cpp

GLM-5.3-Flash (`glm5-next`) support is in llama.cpp pull request
[#27773](https://github.com/ggml-org/llama.cpp/pull/27773) (not merged at the time of writing).
Build that branch:

```sh
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git fetch origin pull/27773/head:glm5next && git checkout glm5next
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j     # CUDA / DGX Spark
# Apple Silicon: cmake -B build && cmake --build build --config Release -j

./build/bin/llama-server -m glm-5.3-flash-e256-q2.gguf -ngl 99 -c 65536 -np 4 --cont-batching -fa on
./build/bin/llama-cli    -m glm-5.3-flash-e256-q2.gguf -ngl 99 -c 32768
```

* Thinking is on by default (the GLM 5.3 template always opens `<think>`); `--reasoning-budget 0`
  turns it off, `--reasoning-budget N` caps it, `--chat-template-kwargs '{"reasoning_effort":"low"}'`
  selects the low/high/max effort levels of the template. The server returns the thinking part
  separately as `reasoning_content`.
* Serving many users: `-np 16 -c 262144` gives 16 slots of 16 K tokens; MoE decode throughput keeps
  growing to 32 slots (table below).
* The file is a plain llama.cpp GGUF (architecture `glm5-next`, standard tensor names, `context_length`
  u32, small mHC/indexer tensors in F32). Two harmless quirks of the branch at the time of writing are
  patched in `llamacpp-pr27773-glm5next-lenient.patch` (tolerate u64 metadata; return unparsed
  replies verbatim instead of HTTP 500 when `max_tokens` cuts a reply mid-UTF-8-character).
* Vision: this is the text model; no `mmproj` is provided yet for the Flash vision encoder under
  llama.cpp.

## Speed (llama.cpp, CUDA, one B200, model resident)

| | tokens/s |
|---|---:|
| Prompt processing (pp512 / pp2048) | 1037 / 1086 |
| Generation, 1 sequence | 54 |
| Generation, 4 / 8 / 16 / 32 parallel sequences (aggregate) | 130 / 170 / 207 / 253 |

(`llama-bench` and `llama-batched-bench`, 256-token prompts, 128 generated tokens each, flash
attention on.) Memory: 79.1 GiB weights + ~1 GiB per 16 K tokens of context; the DSA/KDA layers keep
the KV cache small.

**DGX Spark (128 GB):** the same build (`-DGGML_CUDA=ON`, sm_121) runs the file resident in unified
memory with ~40 GiB left; per-token work is identical to the stock 288-expert Flash Q2, so expect the
same speed class (roughly 15–20 t/s decode) — not measured by the author.

## Quality

Same weights, DwarfStar's `ds4-eval` harness (GPQA Diamond, SuperGPQA, AIME 2025 interleaved; thinking
on, 16 000-token budget, greedy), first 40 core cases: **35 / 40** (3 wrong, 2 out of budget). The 4-bit
model with the same expert selection scored, under vLLM: HumanEval 97.6, C-Eval 89.4, GPQA-Diamond 77.3
(low effort), AIME 2025 74.2, MMMU 76.1, BFCL Non-Live 87.7 / Live 80.5 / multi-turn 73–75. The 2-bit
routed experts add their own loss, mostly on the hardest reasoning. An imatrix-guided build of the
same layout was tested head-to-head and did worse (31 / 40, twice the reasoning tokens), so this
weight-energy-importance build is the one published.

Qualitative: correct bilingual common-sense / medical / coding answers, no Chinese–English mixing.

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
template with tool calling. No MTP block.

## File

| File | Bytes | SHA-256 |
|---|---:|---|
| `glm-5.3-flash-e256-q2.gguf` | 84,929,449,952 | `44ef6ae232e4928803e6b485e165ec3731bebb49fd62ec3e2f4200730295d83a` |

## Limitations

* 2-bit routed experts on top of an 11 % expert pruning: chat, coding and agent use are the target;
  expect a drop on the hardest maths/science reasoning versus the 4-bit vLLM deployment.
* Needs the llama.cpp `glm5-next` branch until it is merged; the DwarfStar fork
  [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim) can also convert the file back to
  its own layout (`gguf-tools/glm5next_to_llamacpp.py --restore`), but llama.cpp is the recommended
  runtime: faster, batched, mainstream.
* No MTP head, no vision projector in this file.

License: MIT (base model). Credits: Z.AI (GLM-5.3-Flash); the llama.cpp / ggml authors and the
`glm5-next` PR authors; DwarfStar (2-bit recipe and quantizer);
[MOE-SLIM](https://github.com/yuhai-china/MOE-SLIM) (expert pruning); yuhai-china (this build).
