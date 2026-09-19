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
---

# GLM-5.3-Flash-E256o — Q2 GGUF for DwarfStar (ds4): GLM 5.3 Flash on a DGX Spark or a 128 GB Mac

**78.9 GiB. GLM-5.3-Flash with 256 of 288 routed experts, routed experts in imatrix-guided
IQ2_XXS (gate/up) + Q2_K (down), everything else Q8_0/BF16 — the same recipe as DwarfStar's
published GLM 5.3 Flash Q2 (90 GiB), 11 GiB smaller, so a 128 GB machine keeps ~40 GiB for context.**

Built from the FP8 checkpoint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
with the expert selection of GLM-5.3-Flash-E256o (the 256-expert NVFP4 model evaluated at HumanEval
97.6 / C-Eval 89.4 / MMMU 76.1 / BFCL Live 80.5 under vLLM); experts were copied byte-exactly from
FP8 and quantized once, not re-quantized from the 4-bit release. Runs with the
[`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim) fork of DwarfStar (upstream pins the
Flash shape to 288 experts + MTP block). No MTP head: `--mtp` is not available.

| File | Size | Importance | Use |
|---|---:|---|---|
| `GLM-5.3-Flash-E256o-Q2-imatrix.gguf` | 84,694,295,648 B (78.9 GiB) | activation imatrix (1.64 M tokens, 10 domains) | **recommended** |
| `GLM-5.3-Flash-E256o-Q2-fallback.gguf` | 84,694,295,648 B | weight-energy heuristic | reference / A-B |

## What is in the file

| Role | Type | Bytes |
|---|---|---:|
| Routed experts gate / up (42 layers × 256) | IQ2_XXS (2.06 bpw) | 46.5 GB |
| Routed experts down | Q2_K (2.63 bpw) | 29.6 GB |
| KDA linear attention, DSA attention, dense FFN, shared experts | Q8_0 (+1.3 GB Q4_K) | 8.2 GB |
| Embedding, output head | BF16 / Q8_0 | 1.3 GB |
| Norms, routers, mHC, indexer | F32 | 0.2 GB |

Architecture unchanged from GLM 5.3 Flash: 45 layers (3 dense + 42 MoE), KDA linear attention with
DSA every fourth layer, hyper-connections, top-8 of 256 routed experts + 1 shared, 154 880-token
vocabulary, 1 M-position RoPE. Vision encoder not included (text GGUF only, as upstream's).

## Running

### DGX Spark (GB10, 128 GB unified memory)

```sh
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4 && cd ds4
make cuda-spark
./ds4 -m /path/to/GLM-5.3-Flash-E256o-Q2-imatrix.gguf --cuda --ctx 32768          # chat
./ds4-server -m /path/to/GLM-5.3-Flash-E256o-Q2-imatrix.gguf --cuda --ctx 65536   # OpenAI API :8000
./ds4-agent  -m /path/to/GLM-5.3-Flash-E256o-Q2-imatrix.gguf --cuda --ctx 65536   # coding agent
```

Expect the startup line `ds4: GLM 5.3 Flash variant: 256 routed experts, 45 blocks, 0 MTP block(s)`.
Memory: 78.9 GiB weights + ~3 GiB graph + KV (KDA layers keep a constant state; the 11 DSA layers
use a compact cache of ~0.05 GiB per 4 K tokens), so 64 K contexts fit comfortably. The Spark path
uses upstream's integrated-memory mapping and the aligned IQ2_XXS/Q2_K kernels for this exact
layout; speed should match upstream's GLM 5.3 Flash Q2 figures on Spark (this build was not run on
a Spark by the author).

### Apple Silicon 128 GB (Metal)

```sh
make
./ds4 -m /path/to/GLM-5.3-Flash-E256o-Q2-imatrix.gguf --ctx 32768
```

### Discrete NVIDIA GPU (measured: one B200)

`make cuda-generic`, then the same commands with `--cuda`. The fork copies the model into VRAM when it
fits (79 GiB here). Measured on a B200: 38 t/s decode, 84 t/s prefill on a 60-token prompt.

Notes: thinking is on by default (`--nothink` for direct answers; the GLM template's
`reasoning_effort` is `low`/`high`/`max`); GLM requires `--power 100` (default); no MTP speculative
decoding.

## Quality

`ds4-eval` probe (first 15 core cases: GPQA Diamond, SuperGPQA, AIME 2025; thinking on, default
budgets), one B200:

| build | passed |
|---|---|
| imatrix | _filled in below_ |
| fallback importance | _filled in below_ |

Qualitative: bilingual common-sense and medical questions answer correctly; Chinese and English do
not mix; code generation is intact (palindrome function with Chinese test cases).

For the underlying 256-expert model's numbers under vLLM (NVFP4, no 2-bit loss) see the
GLM-5.3-Flash-E256o model card: HumanEval 97.6, GPQA-Diamond 77.3 (low), C-Eval 89.4, AIME25 74.2,
MMMU 76.1, BFCL Non-Live 87.7 / Live 80.5 / multi-turn 73–75. The 2-bit routed experts of this file
add their own loss on top; the imatrix build is the one to use.

## How it was built

1. `moe-slim prune zai-org/GLM-5.3-Flash (FP8) --keep 256` with the E256o statistics
   (11 calibration domains, per-file normalised, zh 2 / math 3 / olympiad 2) → identical expert
   selection to the NVFP4 E256o, FP8 experts byte-exact, MTP block dropped.
2. `python -m moe_slim.calib.imatrix` on that checkpoint: per-expert squared input activations
   (gate/up) and squared router-weighted SwiGLU outputs (down) over 1.64 M tokens sampled evenly
   from the ten calibration domains (DwarfStar's own collector is Metal-only).
3. `gguf-tools/glm53_quantize.py --artifact q2 --imatrix … --cuda` (fork): IQ2_XXS gate/up on the
   GPU (byte-identical to the C quantizer), Q2_K down and Q8_0 dense parts on the CPU; 40 minutes.

## Limitations

* 2-bit routed experts on top of an 11 % expert pruning: fine for chat, coding and agent use; expect
  losses on the hardest reasoning versus the 4-bit vLLM deployment.
* Not yet run on a DGX Spark or a Mac by the author; the CUDA discrete-GPU path was.
* Requires the fork; upstream DwarfStar rejects the file (`expected expert_count=288`).
* No MTP head, no vision encoder in this GGUF.

License: MIT (base model). Credits: Z.AI (GLM-5.3-Flash), Salvatore Sanfilippo and the DwarfStar
contributors (`antirez/ds4`, quant recipe and runtime), llama.cpp/GGML (quant formats),
[MOE-SLIM](https://github.com/yuhai-china/MOE-SLIM) (pruning, imatrix), yuhai-china (this build).
