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
| Vision (`--vision` + stock encoder GGUF) | yes | **yes** — same encoder file, tested ([details](#vision-multimodal-use)) |
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

Startup prints `ds4: GLM 5.3 Flash variant: 256 routed experts, 45 blocks, 0 MTP block(s)`. The
model stays in unified memory (no copy); first load is limited by reading 79 GiB from disk, so keep
the file on the internal NVMe. Thinking is on by default — `--nothink` for direct answers, `/think`
and `/nothink` in the chat, and the server honours the GLM `reasoning_effort` field
(`low`/`high`/`max`). Do not pass `--mtp`. For several concurrent users:
`./ds4-server … --ctx 32768 --batched-session 4`. Client setup (Pi, OpenCode, Codex CLI, Claude Code)
is in the fork's `docs/CLIENTS.md`. Image input: see [Vision](#vision-multimodal-use) below.

Also runs on a 128 GB Apple Silicon Mac (`make`, drop `--cuda`) and on discrete NVIDIA GPUs with
≥ 90 GB of VRAM (`make cuda-generic`).

## Vision (multimodal use)

GLM 5.3 Flash is a vision-language model, and this file keeps that: the pruning removed routed
experts only, so the vision tower and the image projector are untouched and DwarfStar's **stock,
unmodified** GLM 5.3 Flash encoder file matches this GGUF. The encoder is a separate 1.1 GB GGUF
(as with the stock Q2); it is not bundled here.

### Setup

```sh
./download_model.sh glm53-vision          # fetches gguf/GLM-5.3-Flash-Vision-Encoder.gguf (1.1 GB)
```

or take the file from the DwarfStar repository `antirez/glm-5.3-flash-gguf`. Memory cost on the
Spark: ~1.1 GB weights plus a few hundred MB while encoding — still ~38 GiB of headroom.

### Chat CLI

```sh
./ds4 -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536 \
      --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf
> /read photo.jpg          # ds4: image 640x400, 345 image tokens -> the model describes it
> 图里的文字是什么？         # follow-up questions refer to the image already in the conversation
```

`/read FILE` with a PNG or JPEG sends the image as a turn of its own (the model responds to it
immediately); text files are read as text. Use several `/read` commands for several images.

### Coding agent

```sh
./ds4-agent -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536 \
            --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf
```

With `--vision` the agent gains the `view_image` tool: it can open screenshots, UI mock-ups, diagrams
or rendered plots in the working directory on its own. Agent sessions containing images cannot yet
be saved with `/save` (upstream limitation).

### Server (OpenAI / Anthropic / Responses APIs)

```sh
./ds4-server -m gguf/glm-5.3-flash-e256-q2.gguf --cuda --ctx 65536 \
             --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf         # :8000
```

Images are sent inline: OpenAI chat and Responses accept PNG/JPEG `data:` URIs, Anthropic accepts
`base64` image sources. Remote URLs and server-side file paths are rejected; up to 16 images per
request, 64 MiB body. Image blocks keep their order relative to text.

```python
import base64, json, urllib.request
img = base64.b64encode(open("chart.png", "rb").read()).decode()
body = {"model": "glm", "temperature": 0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
            {"type": "text", "text": "Read the chart: what is compared and what are the values? /nothink"}]}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

Any OpenAI-compatible client that supports image blocks (Open WebUI, Pi, OpenCode, Cline, …) works the
same way; point it at `http://<spark>:8000/v1`.

### Tips

* `/nothink` (or `reasoning_effort: low`) is usually enough for description, OCR and chart reading;
  keep thinking on for visual maths/diagram reasoning.
* Image cost: about one prompt token per 28×28-pixel block (a 640×400 image ≈ 350 tokens, a
  1-megapixel photo ≈ 1 300), capped at 8 000 tokens (≈ 6 MP; larger photos are downscaled with
  aspect ratio kept). Encoding took ~1–2 s on a B200; expect a few seconds on the Spark. Pre-resizing
  photos to ~1 MP keeps requests fast without hurting OCR of normal text.
* Chinese and English prompts both work; the model answers in the language of the question.

### Tested

On the B200 with this file + stock encoder through `ds4-server`:

| Input | Result |
|---|---|
| Synthetic image: red circle, blue square, caption "DGX SPARK 128GB" (Chinese question) | shapes, colours, positions and text all correct |
| Bar chart, two bars 90 vs 78.9 GiB (English question) | both values read, difference computed as 11.1 GiB / 12.3 % |

The 256-expert model scores **MMMU 76.1** at 4-bit under vLLM (full vision pipeline); the 2-bit
routed experts of this file only affect the language side.

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

Image understanding: see [Vision](#vision-multimodal-use).

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
