---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model:
- autotrust/GLM-5.3-SLIM-E192
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

# GLM-5.3-SLIM-E160 — IQ2_XXS GGUF: GLM 5.3 in 128 GiB, for llama.cpp

**GLM 5.3 (744B MoE) with 160 of its 256 routed experts per layer, routed experts in 2-bit.
127.9 GiB — 22 GiB smaller than the E192 build — so one 180 GB GPU holds it with room for 32
parallel 16 K-token sessions, and a 192 GB Mac Studio runs it comfortably.** Runs in upstream
**llama.cpp** (`glm-dsa` architecture): 43 t/s single-stream, 188 t/s aggregate at 32 parallel
requests on one B200.

Pruned from [`autotrust/GLM-5.3-SLIM-E192`](https://huggingface.co/autotrust/GLM-5.3-SLIM-E192) (itself
192 of 256 experts, no measured loss on coding / cybersecurity / tool calling / maths): the 32
least-used experts per layer were removed after a 3.1 M-token calibration over eight domains
(general, science, Chinese, code, agent, maths, olympiad maths, SWE agent traces). Attention
(MLA + DSA indexer), shared experts, dense layers, router, tokenizer and chat template are
identical to GLM 5.3.

| GLM 5.3 form | Size | Runs on |
|---|---:|---|
| Original FP8 | 756 GB | 8× H200/B200 |
| Full GLM 5.3 IQ2_XXS (`antirez`) | 197 GiB | 256 GB Mac |
| SLIM-E192 IQ2_XXS | 149.7 GiB | one 180 GB GPU; 256 GB Mac (192 GB with small context) |
| **SLIM-E160 IQ2_XXS (this file)** | **127.9 GiB** | **one 180 GB GPU with ~50 GiB for context/batching; 192 GB Mac comfortably** |

## Run it with llama.cpp

```sh
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j      # NVIDIA
# Apple Silicon: cmake -B build && cmake --build build --config Release -j

./build/bin/llama-server -m GLM-5.3-SLIM-E160-IQ2_XXS.gguf -ngl 99 -c 262144 -np 16 --cont-batching -fa on
./build/bin/llama-cli    -m GLM-5.3-SLIM-E160-IQ2_XXS.gguf -ngl 99 -c 32768
```

Thinking is on by default; `--reasoning-budget 0` disables it, `--chat-template-kwargs
'{"reasoning_effort":"low"}'` selects the template's effort level; the server returns reasoning as
`reasoning_content`; tool calling works through the GLM template.

## Speed (llama.cpp, CUDA, one B200, model resident)

| | tokens/s |
|---|---:|
| Prompt processing (pp512 / pp2048) | 760 |
| Generation, 1 sequence | 43 |
| Generation, 2 / 4 / 8 / 16 / 32 parallel sequences (aggregate) | 73 / 97 / 120 / 159 / 188 |

Per-token work equals E192 and the full GLM 5.3 Q2 (8 routed experts per layer); the gain is
memory.

## Quality

Measured on this 2-bit file (one B200; greedy; thinking off unless stated):

| Test | Result | Reference |
|---|---:|---|
| HumanEval (164, greedy, no thinking) | **93.3 %** (153/164) | E192 FP8 with thinking (vLLM): 95.1 |
| C-Eval validation (1 606 questions, direct answer letter, no thinking) | **69.4 %** | thinking-based protocols score higher; this is the fast no-reasoning setting |
| Held-out perplexity vs E192 (8 domains, 262 K tokens, FP8 forward) | **+3.6 %** (3.437 → 3.562); top-1 agreement 88.6 % | E144: +7.4 %, E128: +11.9 %, E112: +18 % |
| GPQA Diamond / SuperGPQA, `ds4-eval` harness, thinking on (partial: 15 cases) | 11 / 15 (GPQA Diamond 7/8, SuperGPQA 4/7; 0 wrong on Diamond) | E192 same harness: Diamond 12/25 (13 out of budget), SuperGPQA 17/25 |

Per-domain PPL change vs E192: maths +0.2 %, olympiad +0.4 %, code +1.6 %, agent traces +3 %,
science +2 %, SWE +4 %, Chinese +8 %, general web text +9 % — the pruning was calibrated towards
code / maths / agent use, which is where it is nearly free.

A per-layer *non-uniform* expert allocation (mass-greedy and layer-sensitivity-weighted variants) was
evaluated at equal budget and gave no measurable gain over uniform counts, so this file keeps 160
experts in every MoE layer.

## What is in the file

| Role | Tensors | Type | Bytes |
|---|---|---|---:|
| Routed experts gate / up / down (75 layers × 160) | 225 | IQ2_XXS | 116.8 GB |
| Attention (MLA q_a/q_b/kv_a/k_b/v_b/o, DSA indexer) | 78 layers | Q8_0 | 14.5 GB |
| Shared experts, dense FFN (layers 0–2) | | Q8_0 | 3.7 GB |
| Token embedding, output head | 2 | Q8_0 | 2.0 GB |
| Norms, routers, bias, indexer weights_proj | — | F32 | 0.4 GB |

78 layers (3 dense + 75 MoE), 64 MLA heads, DSA indexer with top-2048 selection, top-8 of 160
routed experts + 1 shared, 1 M-token positional range, no MTP block. Standard llama.cpp `glm-dsa`
layout (`expert_count = 160`).

## File

```text
GLM-5.3-SLIM-E160-IQ2_XXS.gguf   137,344,279,968 bytes
sha256  728c7ec55242349c230054143c8931ac2f1fb3a0b579ec27d60e205ea4cfc9da
```

## Limitations

* 2-bit routed experts on a 37.5 %-pruned expert set: a measurable drop on knowledge-heavy and
  Chinese-exam tasks versus the FP8 checkpoint; coding, maths and agent use are the intended
  workloads.
* Long chains of thought occasionally loop at greedy decoding (one GPQA Diamond item in the probe);
  use temperature 0.6–1.0 with `--min-p 0.05`, or a reasoning budget.
* No MTP head. Not a 128 GB-machine target (128 GiB weights): for DGX Spark / 128 GB Macs use
  GLM-5.3-Flash-E256 Q2.

## Credits

Z.AI (GLM-5.3) · autotrust (SLIM-E192 pruning) · llama.cpp / ggml · DwarfStar (2-bit recipe, quantizer) ·
[MOE-SLIM](https://github.com/yuhai-china/MOE-SLIM) (calibration and pruning tooling) · yuhai-china (this build).
