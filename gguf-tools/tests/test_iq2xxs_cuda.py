#!/usr/bin/env python3
"""Compare the CUDA IQ2_XXS quantizer with libds4quants byte for byte.

Usage:
  python3 gguf-tools/tests/test_iq2xxs_cuda.py [--hf HF_DIR] [--experts N]

Without --hf only synthetic matrices are checked.  With an FP8 GLM-5.3 style
checkpoint, real routed expert matrices are dequantized once on the CPU (the
reference path) and quantized by both implementations with the same
importance vector.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import glm53_quantize as q  # noqa: E402
import iq2xxs_cuda  # noqa: E402

import torch  # noqa: E402


def dequant_iq2xxs(data, ncols, grid):
    """Reference decode of IQ2_XXS bytes (GGML layout) to float32 values."""
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, 66)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    words = blocks[:, 2:].copy().view(np.uint32).reshape(-1, 8, 2)
    idx = words[:, :, 0]
    aux = words[:, :, 1]
    ls = (aux >> 28).astype(np.float32)
    db = d[:, None] * (0.5 + ls) * 0.25  # [S, 8]
    out = np.zeros((blocks.shape[0], 8, 4, 8), dtype=np.float32)
    for k in range(4):
        gi = (idx >> (8 * k)) & 0xFF
        sign7 = (aux >> (7 * k)) & 0x7F
        parity = np.zeros_like(sign7)
        for b in range(7):
            parity ^= (sign7 >> b) & 1
        signs = sign7 | (parity << 7)
        g = grid[gi]  # [S, 8, 8] values 1/3/5 -> stored grid bytes are 8x that
        sgn = np.where(((signs[:, :, None] >> np.arange(8)) & 1) == 1, -1.0, 1.0).astype(np.float32)
        out[:, :, k, :] = db[:, :, None] * (g * 8).astype(np.float32) * sgn
    return out.reshape(-1, ncols)


def compare(name, x, qw, cpu_quant, gpu_enc, grid):
    t0 = time.time()
    ref = cpu_quant.encode(x, q.QTYPE_IQ2_XXS, qw)
    t1 = time.time()
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(gpu_enc.device)
    got = gpu_enc.encode_to_bytes(xt, qw)
    torch.cuda.synchronize()
    t2 = time.time()
    got_t = gpu_enc.encode(xt, qw)
    torch.cuda.synchronize()
    t3 = time.time()
    assert len(ref) == len(got), (len(ref), len(got))
    a = np.frombuffer(ref, dtype=np.uint8).reshape(-1, 66)
    b = np.frombuffer(got, dtype=np.uint8).reshape(-1, 66)
    same = np.all(a == b, axis=1)
    n = same.size
    n_same = int(same.sum())
    ncols = x.shape[1]
    ya = dequant_iq2xxs(ref, ncols, grid)
    yb = dequant_iq2xxs(got, ncols, grid)
    w = np.broadcast_to(qw[None, :], x.shape)
    err_a = float((w * (ya - x) ** 2).sum())
    err_b = float((w * (yb - x) ** 2).sum())
    print(
        f"{name}: blocks={n} identical={n_same} ({100.0 * n_same / n:.4f}%) "
        f"werr cpu={err_a:.6g} gpu={err_b:.6g} ratio={err_b / err_a if err_a else 0:.6f} "
        f"| C {t1 - t0:.2f}s, CUDA {t3 - t2:.3f}s (first call {t2 - t1:.2f}s)"
    )
    return n, n_same, err_a, err_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", help="FP8 GLM-5.3 checkpoint directory for real expert matrices")
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--library", default=os.path.join(os.path.dirname(HERE), "libds4quants.so"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cpu_quant = q.Quantizer(args.library)
    gpu_enc = iq2xxs_cuda.CudaExpertEncoder()
    grid, _gmap, _nb, counts = iq2xxs_cuda.build_tables()
    print(f"tables: grid={grid.shape} neighbour lists max={counts.max()} min(off-grid)={counts[_gmap < 0].min()}")

    rng = np.random.default_rng(args.seed)
    total = [0, 0]
    # Synthetic: gaussian, heavy tails, sparse zeros, all-negative groups, tiny values, exact zeros.
    cases = {
        "gauss": rng.standard_normal((64, 1024), dtype=np.float32) * 0.02,
        "laplace": rng.laplace(size=(64, 1024)).astype(np.float32) * 0.01,
        "sparse": rng.standard_normal((64, 1024), dtype=np.float32) * (rng.random((64, 1024)) < 0.3),
        "negative": -np.abs(rng.standard_normal((32, 512), dtype=np.float32)),
        "tiny": rng.standard_normal((32, 512), dtype=np.float32) * 1e-9,
        "zeros_mixed": np.where(rng.random((32, 512)) < 0.5, 0.0, rng.standard_normal((32, 512))).astype(np.float32),
        "allzero": np.zeros((8, 256), dtype=np.float32),
    }
    for name, x in cases.items():
        qw = np.square(x, dtype=np.float32).sum(axis=0, dtype=np.float32)
        if name == "allzero":
            qw = np.ones(x.shape[1], dtype=np.float32)
        n, s, _, _ = compare(f"synthetic/{name}", x, qw, cpu_quant, gpu_enc, grid)
        total[0] += n
        total[1] += s
        # also a non-trivial imatrix
        qw2 = (rng.random(x.shape[1]) + 0.05).astype(np.float32)
        n, s, _, _ = compare(f"synthetic/{name}+imatrix", x, qw2, cpu_quant, gpu_enc, grid)
        total[0] += n
        total[1] += s

    if args.hf:
        db = q.SourceDB(args.hf, lambda wm: None)
        config_layers = [3, 20, 41, 60, 77]
        picks = []
        for i in range(args.experts):
            layer = config_layers[i % len(config_layers)]
            part = ("gate", "up", "down")[i % 3]
            expert = int(rng.integers(0, 192))
            picks.append(f"model.layers.{layer}.mlp.experts.{expert}.{part}_proj.weight")
        for name in picks:
            x = cpu_quant.to_f32(db, name)
            qw = np.square(x, dtype=np.float32).sum(axis=0, dtype=np.float32)
            n, s, _, _ = compare(name, x, qw, cpu_quant, gpu_enc, grid)
            total[0] += n
            total[1] += s
            # GPU dequant path must reproduce the CPU dequant exactly
            info = db.info(name)
            codes = db.read(name)
            scales = db.read(name + "_scale_inv")
            xg = gpu_enc.dequantize_fp8(codes, scales, info["shape"]).cpu().numpy()
            if not np.array_equal(xg, x):
                print(f"  !! GPU FP8 dequant differs from CPU for {name}: max abs {np.abs(xg - x).max()}")
                sys.exit(1)
        db.close()

    print(f"TOTAL blocks={total[0]} identical={total[1]} ({100.0 * total[1] / total[0]:.4f}%)")
    if total[1] != total[0]:
        print("mismatching blocks found (see per-case weighted error ratios)")
        sys.exit(2)
    print("OK: CUDA IQ2_XXS output is byte-identical to libds4quants")


if __name__ == "__main__":
    main()
