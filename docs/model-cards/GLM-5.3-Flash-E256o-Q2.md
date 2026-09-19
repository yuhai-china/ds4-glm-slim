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

# GLM-5.3-Flash-E256o Q2 — GLM 5.3 Flash tuned for the DGX Spark

**Built for the 128 GB DGX Spark (GB10): a 78.9 GiB GGUF of GLM-5.3-Flash that leaves ~40 GiB of
unified memory for context and the rest of the system, in the exact tensor layout DwarfStar's Spark
CUDA kernels are optimised for.**

DwarfStar (`ds4`) already runs GLM 5.3 Flash on a Spark with its published Q2 file, but at 90 GiB
that build is "close enough to a 128 GB machine's memory budget that other workloads and context size
matter". This build removes the 32 least-used routed experts per layer first (256 of 288 kept,
selected on a bilingual code / agent / science / maths calibration mix) and then applies the same
Q2 recipe — **IQ2_XXS gate/up + Q2_K down routed experts, Q8_0 everything else** — the layout that
DwarfStar's Spark path (`make cuda-spark`, aligned IQ2_XXS/Q2_K MoE kernels) is built around.
The result is 11 GiB smaller than the stock Q2 with the same per-token speed.

| | Stock DwarfStar GLM 5.3 Flash Q2 | **GLM-5.3-Flash-E256o Q2** |
|---|---|---|
| Size | 90 GiB | **78.9 GiB** |
| Routed experts / layer | 288 | **256** (top-8 active, unchanged) |
| Headroom on a 128 GB Spark (after weights + graph) | ~25 GiB | **~40 GiB** |
| Comfortable context on a Spark | 16–32 K | **64 K+** (KV: ~0.05 GiB per 4 K tokens) |
| Runtime | upstream `antirez/ds4` | [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim) fork (upstream pins 288 experts) |
| MTP speculative decoding | yes | no (MTP block dropped) |
| Importance matrix | yes | two builds: weight-energy (default) and activation imatrix (1.64 M tokens, 10 domains) |

## Files

| File | Bytes | Notes |
|---|---:|---|
| `GLM-5.3-Flash-E256o-Q2-fallback.gguf` | 84,694,295,648 | weight-energy importance — **default** (see Quality) |
| `GLM-5.3-Flash-E256o-Q2-imatrix.gguf` | 84,694,295,648 | activation imatrix (1.64 M tokens, 10 domains); same layout, for A/B |

## Quick start on a DGX Spark

```sh
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4 && cd ds4
make cuda-spark                                   # GB10 build (sm_121)
mkdir -p gguf && mv /path/to/GLM-5.3-Flash-E256o-Q2-fallback.gguf gguf/

./ds4        -m gguf/GLM-5.3-Flash-E256o-Q2-fallback.gguf --cuda --ctx 65536          # chat
./ds4-server -m gguf/GLM-5.3-Flash-E256o-Q2-fallback.gguf --cuda --ctx 65536          # OpenAI API on :8000
./ds4-agent  -m gguf/GLM-5.3-Flash-E256o-Q2-fallback.gguf --cuda --ctx 65536          # native coding agent
```

The first log lines must include `ds4: GLM 5.3 Flash variant: 256 routed experts, 45 blocks, 0 MTP block(s)`.
On the Spark the model stays in unified memory (no copy); startup cost is the 79 GiB read from disk,
so keep the file on the internal NVMe. Thinking is on by default — `--nothink` for direct answers,
`/think` and `/nothink` inside the chat; the GLM template's `reasoning_effort` (`low`/`high`/`max`)
is honoured by the server. `--power 100` (default) is required for GLM. Do not pass `--mtp`.

Serving several users: `./ds4-server … --ctx 32768 --batched-session 4` (context × sessions must
fit; single-GPU CUDA runs the rows as an ordered fallback, i.e. fair scheduling rather than a
throughput multiplier). Client setup for Pi / OpenCode / Codex CLI / Claude Code is in the fork's
`docs/CLIENTS.md`.

### Also runs on

* **128 GB Apple Silicon** (`make`, Metal): same commands without `--cuda`.
* **Discrete NVIDIA GPUs** with ≥ 90 GB (`make cuda-generic`): the fork copies the model into
  VRAM; measured on one B200: 38 t/s decode, 84 t/s prefill on a short prompt.

The Spark and Mac paths are upstream's, exercised by the stock 90 GiB Q2; this file was validated
by the author on the B200 only.

## What changed versus stock GLM 5.3 Flash

* **Experts**: 256 of 288 routed experts per layer, chosen by router-weighted activation mass over
  eleven calibration domains (on-policy GLM traces for general chat, code, tool calling and
  olympiad maths; public science, Chinese, code, agent and maths sets). Attention, KDA layers,
  dense FFN, shared experts, router, tokenizer and chat template are untouched.
* **Quantization**: routed experts IQ2_XXS (gate/up, 2.06 bpw) and Q2_K (down, 2.63 bpw), guided by
  an importance matrix collected on the pruned model; KDA/DSA attention, dense FFN and shared
  experts Q8_0; embeddings/output BF16-Q8; norms/routers F32. Experts were taken byte-exactly from
  the FP8 release and quantized once (not re-quantized from a 4-bit checkpoint).
* **Removed**: the MTP draft block (no speculative decoding) and the vision encoder (text GGUF, as
  upstream's Flash Q2).

| Role | Type | Bytes |
|---|---|---:|
| Routed experts gate / up (42 layers × 256) | IQ2_XXS | 46.5 GB |
| Routed experts down | Q2_K | 29.6 GB |
| KDA + DSA attention, dense FFN, shared experts | Q8_0 (+1.3 GB Q4_K) | 8.2 GB |
| Embedding, output head | BF16 / Q8_0 | 1.3 GB |
| Norms, routers, mHC, indexer | F32 | 0.2 GB |

## Quality

The 256-expert selection itself was evaluated at 4 bit under vLLM (GLM-5.3-Flash-E256o):
HumanEval 97.6, C-Eval 89.4, MMMU 76.1, GPQA-Diamond 77.3 (low effort), AIME 2025 74.2,
BFCL Non-Live 87.7 / Live 80.5 / multi-turn 73–75 — the pruning costs little; the 2-bit routed
experts of this GGUF add their own loss, mostly on the hardest reasoning.

`ds4-eval` (DwarfStar's built-in harness; GPQA Diamond, SuperGPQA, AIME 2025 interleaved; thinking
on, 16 000-token budget, greedy), first 40 core cases on one B200:

| build | passed | wrong | budget exhausted | generated tokens |
|---|---:|---:|---:|---:|
| fallback importance | **35 / 40** | 3 | 2 | 78 k |
| imatrix | 31 / 40 | 4 | 5 | 168 k |

Both builds miss the same four hardest items; the imatrix build additionally loses three GPQA items by
running out of budget (it drifts into 15–16 k-token reasoning loops that the fallback build settles in
1–3 k tokens) and one AIME item, while winning one SuperGPQA item. Forty questions is a coarse probe
and the two are within noise on accuracy alone, but the token-usage pattern is consistent, so the
fallback build is shipped as the default until a perplexity/KL comparison against the FP8 model says
otherwise. Qualitatively both builds answer bilingual common-sense and medical questions correctly,
keep Chinese and English separate, and produce correct code (palindrome function with Chinese test
strings).

## How it was built

1. `moe-slim prune` on the FP8 `zai-org/GLM-5.3-Flash` with the E256o statistics (11 domains,
   per-file normalised; weights zh 2, math 3, olympiad 2, others 1–2) → 256 experts per layer,
   MTP block dropped, 287 GB FP8 checkpoint.
2. `python -m moe_slim.calib.imatrix` on that checkpoint: per-expert squared input activations
   (gate/up) and squared router-weighted SwiGLU outputs (down), 1.64 M tokens sampled evenly from
   the ten calibration domains — DwarfStar's own collector is Metal-only, this reproduces it on CUDA.
3. `gguf-tools/glm53_quantize.py --artifact q2 --imatrix … --cuda` from the
   [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim) fork: IQ2_XXS gate/up on the GPU
   (byte-identical to DwarfStar's C quantizer), Q2_K down and Q8_0 dense parts on the CPU; 40 minutes.

Calibration text (117 k rows, 189 M tokens, all ten domains, verified against the calibration
token streams) and the on-policy OlympiadBench traces are published separately.

## Limitations

* 2-bit routed experts on an 11 %-pruned model: chat, coding and agent use are the target; expect
  a drop on the hardest maths/science reasoning versus the 4-bit vLLM deployment.
* Requires the fork; upstream DwarfStar rejects the file (`expected expert_count=288`).
* No MTP head, no vision encoder.
* Not yet run on a DGX Spark by the author; the code paths are upstream's Spark paths.

License: MIT (base model). Credits: Z.AI (GLM-5.3-Flash); Salvatore Sanfilippo and the DwarfStar
contributors (`antirez/ds4`: runtime, Spark kernels, Q2 recipe); llama.cpp/GGML (quant formats);
[MOE-SLIM](https://github.com/yuhai-china/MOE-SLIM) (pruning, imatrix); yuhai-china (this build).
