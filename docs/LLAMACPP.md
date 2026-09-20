# Running the SLIM / Flash GGUFs with llama.cpp

llama.cpp is the recommended runtime for the GGUFs produced by this fork: it is faster than the
DwarfStar single-GPU path (B200: 42–54 t/s single stream vs 16–38 t/s) and adds continuous batching
(177–253 t/s aggregate at 32 slots), an OpenAI-compatible server with reasoning/tool-call parsing,
and Metal/CUDA/CPU backends.

| File | Architecture | llama.cpp | Notes |
|---|---|---|---|
| GLM-5.3-SLIM-E192 / E160 IQ2_XXS | `glm-dsa` | upstream master | `context_length` must be u32 (`gguf-tools/gguf_fix_ctx_u32.py` fixes DwarfStar-written headers in place) |
| GLM-5.3-Flash-E256 Q2 | `glm5-next` | PR ggml-org/llama.cpp#27773 (unmerged) | DwarfStar layout must be converted once: `gguf-tools/glm5next_to_llamacpp.py IN.gguf --out OUT.gguf` (renames KDA/mHC/indexer tensors, rewrites metadata, `ssm_a = -exp(A_log)`, promotes BF16 mHC/indexer tensors to F32) |

Optional patches in `docs/llamacpp/`:

* `llamacpp-master-glm-dsa-lenient.patch` / `llamacpp-pr27773-glm5next-lenient.patch` — accept
  integer metadata of a different width (u64 `context_length`) and return a reply verbatim instead
  of HTTP 500 when `max_tokens` cuts it in the middle of a UTF-8 sequence.

Measured on one B200 (flash attention on, 256-token prompts, 128 generated tokens):

| Model | pp2048 | tg ×1 | tg ×8 | tg ×32 |
|---|---:|---:|---:|---:|
| SLIM-E192 IQ2_XXS (149.7 GiB) | 730 | 42 | 120 | 177 |
| SLIM-E160 IQ2_XXS (127.9 GiB) | 760 | 43 | 120 | 188 |
| Flash-E256 Q2 (79.1 GiB) | 1086 | 54 | 170 | 253 |

Kernel profile of the single-stream decode (CUPTI): ~1 900 kernels per token; IQ2_XXS expert GEMVs
~20 % of GPU time at ~10 % of HBM bandwidth, Q8_0 dense GEMVs ~29 %, activation quantisation
~7 %, top-k via argsort ~3 % — the headroom for further work (fused top-k, faster IQ2_XXS dot
products, or Q2_K experts at +28 % size).
