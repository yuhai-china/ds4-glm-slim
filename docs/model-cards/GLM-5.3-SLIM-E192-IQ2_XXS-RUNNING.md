# Running GLM-5.3-SLIM-E192 IQ2_XXS with DwarfStar

This guide covers everything needed to run `GLM-5.3-SLIM-E192-IQ2_XXS.gguf`
(149.7 GiB) on a Mac or an NVIDIA machine with the `ds4-glm-slim` fork of
DwarfStar: building the runtime, planning memory, starting the CLI, the
OpenAI-compatible server and the coding agent, and fixing the errors you are
most likely to meet. Read the [model card](README.md) first for what the file
is and what to expect from it.

Contents

1. [Which runtime](#1-which-runtime)
2. [Build DwarfStar (fork)](#2-build-dwarfstar-fork)
3. [Get and verify the model](#3-get-and-verify-the-model)
4. [Memory planning](#4-memory-planning)
5. [Apple Silicon (Metal)](#5-apple-silicon-metal)
6. [NVIDIA (CUDA)](#6-nvidia-cuda)
7. [Using it: CLI, server, agent](#7-using-it-cli-server-agent)
8. [Thinking, sampling and GLM specifics](#8-thinking-sampling-and-glm-specifics)
9. [Evaluate](#9-evaluate)
10. [Troubleshooting](#10-troubleshooting)
11. [Rebuilding or re-quantizing](#11-rebuilding-or-re-quantizing)

---

## 1. Which runtime

The file is a DwarfStar GLM-DSA GGUF. It needs the **`ds4-glm-slim` fork**
(branch `glm53-slim`), not upstream `antirez/ds4` and not llama.cpp / Ollama /
LM Studio / MLX.

| Runtime | Loads this file? |
|---|---|
| [`ds4-glm-slim`](https://github.com/yuhai-china/ds4-glm-slim) fork, Metal | yes (upstream code paths; the fork only relaxes the shape check) |
| `ds4-glm-slim` fork, CUDA | yes (fork adds IQ2_XXS-down kernels and resident weights) |
| `ds4-glm-slim` fork, ROCm | shape check passes; routed kernels untested |
| upstream `antirez/ds4` | no — `expected expert_count=256 for GLM 5.2, got 192` |
| llama.cpp and derivatives | no — different tensor layout and metadata |
| MLX | no — MLX uses its own safetensors quantization, not GGUF |

The fork is upstream DwarfStar `6289c51` (September 2026) plus a handful of
commits. It keeps running the official GLM 5.2/5.3 and DeepSeek files.

## 2. Build DwarfStar (fork)

### macOS (Apple Silicon)

```sh
xcode-select --install        # once, if the command-line tools are missing
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4
cd ds4
make
./ds4 --help | head -3
```

The same binary serves M2/M3/M4/M5 machines. Nothing GLM-specific has to be
enabled; the Metal kernels are compiled at first start.

### Linux + NVIDIA

Needs the NVIDIA driver, a CUDA toolkit with `nvcc` and cuBLAS (12.x or 13.x),
and a C compiler.

```sh
git clone https://github.com/yuhai-china/ds4-glm-slim.git ds4
cd ds4
make cuda-generic             # -arch=native: builds for the GPU in the machine
# or an explicit architecture, e.g. Blackwell B200:
make cuda CUDA_ARCH=sm_100
```

`make cuda-generic` compiles `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`
and `ds4-eval`. The full build takes 5–10 minutes (the CUDA translation unit
is large).

### Check the build recognises the file

Start it once with a tiny context; the first log lines must include

```text
ds4: GLM DSA variant: 192 routed experts, 78 blocks, 0 MTP block(s)
```

If instead you see `expected expert_count=256 for GLM 5.2, got 192`, the
binary is upstream DwarfStar or an old checkout of the fork.

## 3. Get and verify the model

Place the file where you like; the examples use `gguf/` inside the repository
(`download_model.sh` uses the same directory for the official models).

```sh
mkdir -p gguf
# copy / download GLM-5.3-SLIM-E192-IQ2_XXS.gguf into gguf/
ls -l gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf     # 160760301792 bytes
shasum -a 256 gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf
# expected: 68abcb6effe4e7a4379d92f68607dfbd1aca0a1fad5dabda78a96f1a4a0e4725
```

Keep it on a fast local SSD. Resident runs read the whole file once at start
(150 GiB: a few minutes from a fast NVMe, longer from Thunderbolt or network
storage); SSD-streaming runs read from it continuously.

## 4. Memory planning

What the runtime needs, roughly:

| Component | Size |
|---|---|
| Weights (resident) | 149.7 GiB |
| Graph scratch, prefill buffers | ~4–5 GiB (grows a little with the prefill chunk) |
| KV cache (compact DSA, f16) | ~95 KB per token → 0.7 GiB at 8K, 2.9 GiB at 32K, 5.8 GiB at 64K, 11.7 GiB at 128K |
| Operating system and other apps (Mac) | whatever you leave it |

DwarfStar has a **memory guard** for GLM that refuses a configuration which
cannot fit *before* allocating. Its budget is `physical × 0.99 − reserve`. The
reserve defaults to 32 GiB on machines with more than 160 GiB (this is the
value tuned for unified-memory Macs); on discrete CUDA GPUs the fork uses
`memory/16` (≥ 6 GiB). Override with `DS4_GLM_MEMORY_GUARD_RESERVE_GB=N`, or
disable with `DS4_GLM_MEMORY_GUARD=0` — only when you understand the numbers
above, because an out-of-memory failure on a Mac can freeze the machine.

Recommended starting points:

| Machine | Mode | Notes |
|---|---|---|
| Mac Studio 512 GB | resident | any context, several server sessions |
| Mac Studio 256 GB | resident | `--ctx 65536` and more; leaves ~90 GiB for macOS/apps |
| Mac Studio 192 GB | resident, tight | ~178 GiB physical; weights + graph ≈ 155 GiB; needs `DS4_GLM_MEMORY_GUARD_RESERVE_GB=16`, `--ctx 8192–16384`, no other heavy apps. Otherwise use `--ssd-streaming`. |
| Mac 128 GB (M5 Max, M4 Max) | `--ssd-streaming` | ~4–6 t/s class generation for this size of model (see §5) |
| Mac 64/96 GB | not recommended | streaming works in principle but most experts miss the cache |
| 1× NVIDIA 180 GB (B200, GB200) | resident | 21 t/s decode measured; 8 GiB headroom rule |
| 1× NVIDIA 141 GB (H200) / 80 GB | not resident | would stream experts over PCIe at ~1 t/s; not recommended |
| 2× NVIDIA (e.g. 2× 96 GB) | `--gpu-devices 0,1 --gpu-vram auto` | layer placement across GPUs; not tested with this file |

## 5. Apple Silicon (Metal)

### Resident (256 GB and up)

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 65536
```

The startup report prints the memory plan (`memory: KV … + buffers … + resident
model 149.71 GiB`). The first load reads 150 GiB from disk; later loads are
faster while the file is in the page cache.

### Resident on 192 GB

```sh
DS4_GLM_MEMORY_GUARD_RESERVE_GB=16 ./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 8192
```

Quit browsers and IDEs first. If macOS starts swapping (watch Activity
Monitor's memory pressure), fall back to streaming. This configuration was not
exercised by the author of this file.

### SSD streaming (128 GB)

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ssd-streaming --ctx 8192
```

* Start with the automatic cache budget. The report shows the effective expert
  cache and resident layers.
* Pruned variants start **cold**: the built-in GLM expert hot seed refers to
  the original 256-expert numbering and is skipped, so the cache fills on
  demand during the first prompts. Expect the first hundreds of tokens to be
  slower than steady state.
* Reference from upstream's measurements on a 128 GB M5 Max with the 197 GiB
  full GLM 5.3 IQ2_XXS: 4–5 t/s generation at 8K context with a ~61 GiB expert
  cache. This file has 25 % fewer routed bytes, so a larger share of the
  experts stays cached; expect the same or better.
* `--ssd-streaming-cache-experts 40GB` trades cache for context/sessions;
  `--ssd-streaming-full-layers N` pins the first N routed layers instead of
  spending the budget on selected experts. See `docs/SSD_STREAMING.md`.
* Use a fast internal or Thunderbolt 4/5 NVMe SSD.

### Two Macs

Upstream supports a 50/50 routed-expert split across two 128 GB Macs over
Thunderbolt RDMA or TCP for IQ2_XXS gate/up layouts (`docs/DISTRIBUTED.md`).
This file has not been tried in that mode.

## 6. NVIDIA (CUDA)

### Single GPU with ≥ 158 GiB free (B200 180 GB)

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --ctx 65536
```

What you should see:

```text
ds4: GLM DSA variant: 192 routed experts, 78 blocks, 0 MTP block(s)
ds4: CUDA backend initialized on NVIDIA B200 (sm_100) dev=0
ds4: CUDA copying 149.72 GiB model to device memory (single discrete GPU with room; …)
ds4: CUDA model copy complete in 21.6s
…
ds4: GLM prefill: 43.76 t/s, generation: 21.19 t/s
```

The fork copies the whole model into VRAM when one non-integrated GPU has
`model + 8 GiB` free. Without that line the weights stay in a pinned host
mapping and every token streams the routed experts over PCIe (~1 t/s).
`DS4_CUDA_COPY_MODEL=1` forces the copy, `DS4_CUDA_NO_MODEL_COPY=1` prevents it.

Measured on a B200 (this file, `--nothink`, greedy): 21 t/s decode; 44 t/s
prefill on a 30-token prompt where fixed per-layer cost dominates. Long prompts
use the mmq GEMM tier and prefill considerably faster per token.

VRAM budget on a 180 GB card: 149.7 GiB weights + ~5 GiB graph + f16 copies of
the Q8_0 dense weights that cuBLAS uses (~10–15 GiB) + KV. `--ctx 65536` fits;
`--ctx 131072` is borderline. `--gpu-vram` is not needed for a single GPU.

### DGX Spark / GB10 (integrated, 128 GB)

`make cuda-spark`, then `--cuda --ssd-streaming`: unified memory holds the
19.2 GiB of non-expert weights, the context and a bounded expert cache; the
remaining experts are read from NVMe per token.

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --ssd-streaming --ctx 16384
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --ssd-streaming --ssd-streaming-cache-experts 64GB --ctx 16384
```

Plan printed by ds4 for a 40 GiB cache on our CUDA box (the same layout applies
on a Spark): `resident model 19.19 GiB + expert cache 36.52 GiB (4029 experts,
9.28 MiB each) + prefill expert reserve 3.48 GiB + KV 0.36 GiB + buffers 4 GiB`.
On a 128 GB Spark a 64–72 GiB cache (≈ 7,000–7,700 of 14,400 expert slots) keeps
the total near 95–100 GiB. Generation is NVMe-bound: low single-digit t/s once
warm, slower while the cache fills (pruned files start cold). `--nothink` and
a 16–32 K context keep paging down. Verified with this file on a B200 in
streaming mode (correct output, 40 GiB cache, cold start); not run on a Spark
by the author.

### Several GPUs

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --cuda --gpu-devices 0,1 --gpu-vram auto --ctx 32768
```

Whole layers are placed per device; the resident copy is not used in this
mode (each device caches its own layers). Untested with this file; the routed
kernels are the same as on one GPU.

## 7. Using it: CLI, server, agent

Run the binaries from the repository root, or pass `--chdir /path/to/ds4` so
the Metal kernels and runtime files are found.

### Interactive CLI

```sh
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 32768          # Metal
./ds4 -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 32768 --cuda   # CUDA
./ds4 -m … -p "Explain Redis streams in one paragraph." --nothink   # one shot
```

Inside the chat: `/help`, `/read FILE`, `/ctx N`, `/nothink` and `/think`,
`/quit`. Ctrl+C interrupts a generation.

### OpenAI-compatible server

```sh
./ds4-server -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 65536 [--cuda] [--host 0.0.0.0]
```

Endpoints on `http://127.0.0.1:8000`: `GET /v1/models`,
`POST /v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/messages`
(Anthropic style). Tools and SSE streaming are supported; reasoning comes back
in each API's native field.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.3-slim",
    "messages": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    "max_tokens": 512,
    "stream": false
  }'
```

The `model` string is an alias; the GGUF passed at startup is what runs.

Several concurrent sessions: `--batched-session 4` preallocates four KV
states (context × 4 must fit). On single-GPU CUDA and on Metal GLM 5.2-class
models the rows are executed in an ordered fallback: fair scheduling, not a
throughput multiplier.

Client setups for Pi, OpenCode, Codex CLI and Claude Code are in
`docs/CLIENTS.md`; disk-backed KV caches for prompt reuse are in
`docs/SERVER.md`.

### Coding agent

```sh
./ds4-agent -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 65536 [--cuda]
```

Runs inference in-process with GLM's native tool format, keeps sessions in
`~/.ds4/kvcache` (`/save`, `/list`, `/switch <sha>`, `/del`, `/strip`), and can
resume compatible KV snapshots without re-prefilling.

## 8. Thinking, sampling and GLM specifics

* **Thinking is on by default.** `--nothink` (or `/nothink`) for direct
  answers; `--think` re-enables it. In the server use `think:false` / a
  disabled `reasoning` object, or `reasoning_effort`.
* **Sampling defaults for GLM**: temperature 1.0, top-p 0.95, min-p 0 in the
  CLI/agent; the server uses temperature 1, top-p 1, min-p 0.05 unless the
  request sets them. `--temp 0` is greedy.
* **`--power 100` is required** for GLM (the default).
* **No `--mtp`.** The checkpoint ships no MTP head
  (`glm-dsa.nextn_predict_layers = 0`); speculative decoding is unavailable.
* **Prefill chunking** is graph-selected for GLM; `--prefill-chunk` is not
  accepted.
* **Chinese** works (the tokenizer and template are GLM-5.3's); knowledge-exam
  style Chinese questions are the weak spot of the pruned base (C-Eval −7 pt)
  and the 2-bit experts add to that.
* **Context**: the metadata allows 1M positions. Memory is the limit; with
  `--ctx` above ~128K also expect slower prefill (dense attention limit 2048
  with DSA top-k selection beyond).

## 9. Evaluate

DwarfStar's built-in harness (GPQA Diamond, SuperGPQA, AIME 2025; multiple
choice and integer answers, thinking on):

```sh
./ds4-eval -m gguf/GLM-5.3-SLIM-E192-IQ2_XXS.gguf --ctx 32768 --plain --suite core \
  --trace eval-core.txt [--cuda]
./ds4-eval … --suite hard --retry-incomplete      # 50 harder cases
./ds4-eval … --questions 5                        # quick probe
./ds4-eval --regrade-trace eval-core.txt          # re-score without the model
```

Default generation budgets are 4K–16K tokens per case; the core suite takes
1–3 hours at ~20 t/s. `ds4-eval` is an integration check for the runtime and
the quant, not a leaderboard. For A/B comparisons run the same suite with the
published full GLM 5.3 Q2 on the same machine (it needs `--ssd-streaming` on
anything under 256 GB).

`gguf-tools/quality-testing/` has the continuation scorer used upstream for
quant A/B against official API completions; it needs reference data collected
for this model first.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ds4: expected expert_count=256 for GLM 5.2, got 192` | upstream ds4 or an old fork checkout | build the `glm53-slim` branch of the fork |
| `ds4: GLM prefill failed` on CUDA right after `attn_output` of layer 3 | fork older than commit `64694dd` (no IQ2_XXS-down kernels) | update and rebuild (`make cuda-generic`) |
| `ds4: routed moe: unsupported routed expert types gate/up=16 down=16` | same as above | same |
| `ds4: GLM memory guard refused ctx=… required model+graph: 154.5 GiB; guard budget: …` | reserve too large for this machine (192 GB Mac), or the context too big | `DS4_GLM_MEMORY_GUARD_RESERVE_GB=16`, smaller `--ctx`, or `--ssd-streaming`; do not just disable the guard on a Mac |
| CUDA: ~1 token/s, GPU memory used ≈ 18 GB | weights left in the pinned host mapping | make sure the log shows `CUDA copying … model to device memory`; free VRAM must be ≥ 158 GiB; unset `DS4_CUDA_NO_MODEL_COPY`, `DS4_CUDA_WEIGHT_CACHE`, `DS4_CUDA_WEIGHT_PRELOAD` |
| CUDA: `CUDA model allocation skipped: out of memory` | another process holds VRAM (`nvidia-smi`) | stop it (e.g. a vLLM server) or accept the slow mapping |
| Garbled or repetitive output | wrong file or damaged download | verify size and sha256; try `--temp 0` on a factual prompt |
| `--mtp` does nothing / is rejected | no MTP head in this checkpoint | run without it |
| Mac: the machine becomes unresponsive on load | memory oversubscription | stop other apps, lower `--ctx`, or use `--ssd-streaming`; never combine `DS4_GLM_MEMORY_GUARD=0` with a resident load on 192 GB |
| SSD streaming very slow for the first prompt | cold cache (no hot seed for pruned variants) | expected; steady state improves; a warm-up prompt helps; `--ssd-streaming-cache-experts` larger if memory allows |
| `ds4-server`: request refused for memory | context × sessions does not fit | lower `--ctx` or `--batched-session` |

Useful diagnostics: `DS4_GLM_TP_DEBUG=1` prints why a routed MoE path bailed
out; `nvidia-smi` should show ~155–170 GB used on a resident CUDA run;
`ds4 --help runtime` lists the memory and streaming flags.

## 11. Rebuilding or re-quantizing

The fork's `docs/GLM53_SLIM.md` and `gguf-tools/README.md` document the
pipeline:

* `gguf-tools/glm53_full_quantize.py --hf <FP8 checkpoint> --tokenizer-template
  <GLM GGUF> --cuda` reproduces this file (about 30 minutes on one B200; the
  CPU path takes hours). Any GLM 5.x DwarfStar GGUF works as tokenizer
  template — this file itself, or a 9 MiB tokenizer-only GGUF made with
  `gguf-tools/gguf_tokenizer_only.py`.
* `--dry-run` prints the exact size and validates the checkpoint and tokenizer
  without writing anything.
* An imatrix pass: run this GGUF through `ds4 --imatrix-dataset
  gguf-tools/imatrix/dataset/rendered_prompts.txt --imatrix-out slim.dat`, then
  quantize again with `--imatrix slim.dat`. The CUDA quantizer is
  byte-identical to the C one for a given importance vector, so the only
  difference between a GPU and a CPU build is the fallback importance summation
  order when no imatrix is supplied.
* Recipe variants (Q2_K down projections, `--provisional-q2k`) are one flag
  away; sizes: all-IQ2_XXS 149.7 GiB, IQ2_XXS gate/up + Q2_K down ≈ 162 GiB,
  all-Q2_K ≈ 185 GiB.
