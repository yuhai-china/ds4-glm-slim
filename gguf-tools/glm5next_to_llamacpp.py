#!/usr/bin/env python3
"""Rewrite a DwarfStar GLM-5.3-Flash ("glm5-next") GGUF header in place so that
llama.cpp's GLM5-Next implementation (PR ggml-org/llama.cpp#27773) loads it.

Only the header (metadata + tensor names) is rewritten; tensor data stays where it
is, except the KDA A_log vectors which are converted in place to llama.cpp's
convention ssm_a = -exp(A_log).  The rewritten header is padded with a
`general.padding` string so that the data section offset is unchanged.

Usage:  glm5next_to_llamacpp.py MODEL.gguf [--dry-run] [--backup HEADER.bak]
        glm5next_to_llamacpp.py MODEL.gguf --restore HEADER.bak   (undo; A_log restored too)
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

ALIGN_DEFAULT = 32
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR_FMT = {T_U8: "B", T_I8: "b", T_U16: "H", T_I16: "h", T_U32: "I", T_I32: "i", T_F32: "f", T_BOOL: "?",
              T_U64: "Q", T_I64: "q", T_F64: "d"}

RENAMES = {
    "kda_q.weight": "attn_q.weight",
    "kda_k.weight": "attn_k.weight",
    "kda_v.weight": "attn_v.weight",
    "kda_q_conv.weight": "ssm_conv1d_q.weight",
    "kda_k_conv.weight": "ssm_conv1d_k.weight",
    "kda_v_conv.weight": "ssm_conv1d_v.weight",
    "kda_f_a.weight": "ssm_f_a.weight",
    "kda_f_b.weight": "ssm_f_b.weight",
    "kda_dt_bias.weight": "ssm_dt.bias",
    "kda_a_log.weight": "ssm_a",
    "kda_beta.weight": "ssm_beta.weight",
    "kda_g_a.weight": "ssm_g_a.weight",
    "kda_g_b.weight": "ssm_g_b.weight",
    "kda_o_norm.weight": "ssm_norm.weight",
    "kda_output.weight": "attn_output.weight",
    "indexer.pool_ape.weight": "indexer_compressor_ape.weight",
    "indexer.pool_gate.weight": "indexer_compressor_gate.weight",
}
DROP_KEYS = {
    "glm5-next.trunk_block_count", "glm5-next.attention.rope_dimension_count",
    "glm5-next.attention.indexer.pool_size", "glm5-next.linear_attention.head_count",
    "glm5-next.linear_attention.head_dimension", "glm5-next.linear_attention.conv_kernel",
    "glm5-next.linear_attention.gate_lower_bound", "glm5-next.layer_types", "glm5-next.swiglu_limit",
    "glm5-next.attention.key_length", "glm5-next.attention.value_length", "general.padding",
    "general.source.revision",
}


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.p)[0]
        self.p += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.take("Q")
        s = self.b[self.p:self.p + n].decode("utf-8")
        self.p += n
        return s

    def value(self, t):
        if t == T_STR:
            return self.string()
        if t == T_ARR:
            et = self.take("I")
            n = self.take("Q")
            return (et, [self.value(et) for _ in range(n)])
        return self.take(SCALAR_FMT[t])


def pack_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def pack_value(t, v):
    if t == T_STR:
        return pack_string(v)
    if t == T_ARR:
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(pack_value(et, x) for x in items)
    return struct.pack("<" + SCALAR_FMT[t], v)


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(64 * 1024 * 1024)   # generous: GLM headers are ~10 MB
    r = Reader(head)
    if r.take("4s") != b"GGUF":
        sys.exit("not a GGUF file")
    version = r.take("I")
    n_tensors = r.take("Q")
    n_kv = r.take("Q")
    kvs = []
    for _ in range(n_kv):
        k = r.string()
        t = r.take("I")
        kvs.append([k, t, r.value(t)])
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        typ = r.take("I")
        off = r.take("Q")
        tensors.append([name, dims, typ, off])
    header_end = r.p
    align = next((v for k, t, v in kvs if k == "general.alignment"), ALIGN_DEFAULT)
    data_start = (header_end + align - 1) // align * align
    return version, kvs, tensors, header_end, data_start, align


def serialize(version, kvs, tensors):
    out = [b"GGUF", struct.pack("<IQQ", version, len(tensors), len(kvs))]
    for k, t, v in kvs:
        out.append(pack_string(k) + struct.pack("<I", t) + pack_value(t, v))
    for name, dims, typ, off in tensors:
        out.append(pack_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                   + struct.pack("<IQ", typ, off))
    return b"".join(out)


def convert(kvs, tensors):
    kv = {k: (t, v) for k, t, v in kvs}
    arch = kv["general.architecture"][1]
    if arch != "glm5-next":
        sys.exit(f"architecture is {arch}, expected glm5-next")
    if "glm5-next.kda.head_dim" in kv:
        sys.exit("already converted (glm5-next.kda.head_dim present)")
    layer_types = kv["glm5-next.layer_types"][1][1]          # 1 = MLA/DSA layer, 0 = KDA layer
    n_layer = kv["glm5-next.block_count"][1]
    if len(layer_types) != n_layer:
        sys.exit("layer_types length mismatch")
    conv_kernel = kv["glm5-next.linear_attention.conv_kernel"][1]
    kda_head_dim = kv["glm5-next.linear_attention.head_dimension"][1]
    gate_lb = kv["glm5-next.linear_attention.gate_lower_bound"][1]
    kv_lora = kv["glm5-next.attention.kv_lora_rank"][1]
    key_len_mla = kv["glm5-next.attention.key_length"][1]     # 256 (nope only)
    val_len_mla = kv["glm5-next.attention.value_length"][1]   # 256
    pool = kv["glm5-next.attention.indexer.pool_size"][1]
    swiglu = kv.get("glm5-next.swiglu_limit", (T_F32, None))[1]

    new_kvs = [[k, t, v] for k, t, v in kvs if k not in DROP_KEYS]
    def put(k, t, v):
        for e in new_kvs:
            if e[0] == k:
                e[1], e[2] = t, v
                return
        new_kvs.append([k, t, v])
    put("glm5-next.attention.head_count_kv", T_ARR, (T_U32, [1 if lt else 0 for lt in layer_types]))
    put("glm5-next.attention.key_length", T_U32, kv_lora)          # rope dim is 0: kv_lora + 0
    put("glm5-next.attention.value_length", T_U32, kv_lora)
    put("glm5-next.attention.key_length_mla", T_U32, key_len_mla)
    put("glm5-next.attention.value_length_mla", T_U32, val_len_mla)
    put("glm5-next.rope.dimension_count", T_U32, 0)
    put("glm5-next.attention.layer_norm_epsilon", T_F32, 1e-6)
    put("glm5-next.ssm.conv_kernel", T_U32, conv_kernel)
    put("glm5-next.kda.head_dim", T_U32, kda_head_dim)
    put("glm5-next.kda.gate_lower_bound", T_F32, gate_lb)
    put("glm5-next.attention.indexer.kpool", T_U32, pool)
    put("glm5-next.attention.indexer.kpool_select_tail", T_BOOL, True)
    put("glm5-next.expert_gating_func", T_U32, 2)                    # sigmoid
    if swiglu is not None:
        put("glm5-next.swiglu_clamp_exp", T_F32, float(swiglu))
        put("glm5-next.swiglu_clamp_shexp", T_F32, float(swiglu))
    # context_length as u32 for stock loaders
    for e in new_kvs:
        if e[0] == "glm5-next.context_length" and e[1] == T_U64:
            e[1] = T_U32
    # indexer.types is left out: the loader defaults every layer to a full indexer, which is
    # what GLM-5.3-Flash has (every DSA layer owns its indexer); dropping it keeps the header short.

    new_tensors = []
    a_log = []
    for name, dims, typ, off in tensors:
        prefix, sep, local = name.partition(".")
        if prefix == "blk":
            li, sep2, local2 = local.partition(".")
            if local2 in RENAMES:
                if local2 == "kda_a_log.weight":
                    a_log.append((off, dims[0]))
                local2 = RENAMES[local2]
            name = f"blk.{li}.{local2}"
        new_tensors.append([name, dims, typ, off])
    return new_kvs, new_tensors, a_log


def fit_padding(version, kvs, tensors, data_start, align):
    """Add general.padding so that align_up(header_end) == data_start."""
    base = serialize(version, kvs, tensors)
    if len(base) > data_start:
        sys.exit(f"new header ({len(base)} B) is longer than the original data offset ({data_start} B); drop more keys")
    # padding record: key(8+15) + type(4) + len(8) + payload
    fixed = 8 + len(b"general.padding") + 4 + 8
    target_lo = data_start - align + 1
    need = max(0, target_lo - len(base))
    payload = max(0, need - fixed)
    for extra in range(0, align + 1):
        cand = kvs + [["general.padding", T_STR, "x" * (payload + extra)]]
        blob = serialize(version, cand, tensors)
        end = (len(blob) + align - 1) // align * align
        if end == data_start and len(blob) <= data_start:
            return blob
    # if the base header already lands on the same aligned boundary, no padding needed
    end = (len(base) + align - 1) // align * align
    if end == data_start:
        return base
    sys.exit("could not fit the padded header")


TYPE_BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 10: (256, 84), 12: (256, 144), 16: (256, 66),
              14: (256, 210), 13: (256, 176), 11: (256, 110), 24: (1, 1), 26: (1, 4)}
PROMOTE = ("hc_attn_fn.weight", "hc_ffn_fn.weight", "indexer.attn_q_b.weight", "indexer.attn_k.weight",
           "indexer.proj.weight", "indexer_compressor_ape.weight", "indexer_compressor_gate.weight")


def tensor_nbytes(dims, typ):
    blk, tsz = TYPE_BLOCK[typ]
    return math.prod(dims) // blk * tsz


def rewrite(src, dst):
    """Full repack: new header + tensors copied in order (llama.cpp requires packed offsets)."""
    version, kvs, tensors, header_end, data_start, align = read_header(src)
    new_kvs, new_tensors, a_log_set = convert(kvs, tensors)
    a_log_offsets = {off for off, n in a_log_set}
    for e in new_kvs:
        if e[0] == "general.padding":
            new_kvs.remove(e)
    plan = []          # (name, dims, new_type, new_off, src_off, src_type)
    cur = 0
    for name, dims, typ, off in new_tensors:
        local = name.split(".", 2)[-1] if name.startswith("blk.") else name
        new_typ = 0 if (typ == 30 and local in PROMOTE) else typ
        nbytes = tensor_nbytes(dims, new_typ)
        plan.append([name, dims, new_typ, cur, off, typ])
        cur += (nbytes + align - 1) // align * align
    blob = serialize(version, new_kvs, [[n, d, t, o] for n, d, t, o, _, _ in plan])
    pad = (-len(blob)) % align
    total = len(blob) + pad + cur
    print(f"rewriting -> {dst}: {len(plan)} tensors, {sum(1 for p_ in plan if p_[2] != p_[5])} promoted to F32, "
          f"{len(a_log_offsets)} A_log converted, {total / 1e9:.2f} GB")
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(blob)
        fo.write(b"\0" * pad)
        for i, (name, dims, new_typ, new_off, src_off, src_typ) in enumerate(plan):
            fi.seek(data_start + src_off)
            n_src = tensor_nbytes(dims, src_typ)
            if new_typ != src_typ:                                   # BF16 -> F32
                raw = np.frombuffer(fi.read(n_src), dtype="<u2").astype(np.uint32) << 16
                fo.write(raw.view("<f4").astype("<f4").tobytes())
            elif src_off in a_log_offsets:
                a = np.frombuffer(fi.read(n_src), dtype="<f4").astype(np.float64)
                fo.write((-np.exp(a)).astype("<f4").tobytes())
            else:
                remaining = n_src
                while remaining:
                    chunk = fi.read(min(remaining, 256 << 20))
                    fo.write(chunk)
                    remaining -= len(chunk)
            nbytes = tensor_nbytes(dims, new_typ)
            fo.write(b"\0" * ((-nbytes) % align))
            if i % 200 == 0:
                print(f"  {i}/{len(plan)} {name}", flush=True)
    print(f"done: {os.path.getsize(dst)} bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", help="write the original header bytes here (default: MODEL.header.bak)")
    ap.add_argument("--restore", help="restore this header backup (and un-convert A_log)")
    ap.add_argument("--out", help="write a new, fully repacked GGUF here instead of editing in place "
                                  "(BF16 mHC/indexer tensors promoted to F32, tensors packed in order)")
    args = ap.parse_args()
    if args.out:
        rewrite(args.model, args.out)
        return
    version, kvs, tensors, header_end, data_start, align = read_header(args.model)
    print(f"{args.model}: gguf v{version}, {len(tensors)} tensors, {len(kvs)} kv, header {header_end} B, data at {data_start}")

    if args.restore:
        with open(args.restore, "rb") as f:
            orig = f.read()
        if len(orig) != data_start:
            sys.exit("backup length does not match the data offset")
        # un-convert A_log: current file has ssm_a = -exp(A_log)
        kv = {k: (t, v) for k, t, v in kvs}
        ssm = [(off, dims[0]) for name, dims, typ, off in tensors if name.endswith(".ssm_a")]
        with open(args.model, "r+b") as f:
            for off, n in ssm:
                f.seek(data_start + off)
                a = np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float64)
                f.seek(data_start + off)
                f.write(np.log(-a).astype("<f4").tobytes())
            f.seek(0)
            f.write(orig)
        print(f"restored header and {len(ssm)} A_log tensors (appended F32 copies at the end of the file are unreferenced now; truncate with the original size if needed)")
        return

    new_kvs, new_tensors, a_log = convert(kvs, tensors)
    blob = fit_padding(version, new_kvs, new_tensors, data_start, align)
    renamed = sum(1 for (a, *_), (b, *_) in zip(tensors, new_tensors) if a != b)
    print(f"renamed {renamed} tensors, {len(new_kvs)} kv records, new header {len(blob)} B (data offset unchanged), "
          f"{len(a_log)} A_log vectors to convert")
    if args.dry_run:
        for k, t, v in new_kvs:
            if k.startswith("glm5-next.") and not isinstance(v, tuple):
                print(f"  {k} = {v}")
            elif k.startswith("glm5-next.") and isinstance(v, tuple):
                print(f"  {k} = array[{len(v[1])}] {v[1][:12]}...")
        return
    backup = args.backup or args.model + ".header.bak"
    with open(args.model, "rb") as f:
        orig = f.read(data_start)
    with open(backup, "wb") as f:
        f.write(orig)
    # llama.cpp keeps the small mHC / indexer / compressor parameters in F32; DwarfStar
    # stores them in BF16, which the binary ops (e.g. the k-pool APE add) do not accept.
    # Promote them: append F32 copies at the end of the file and repoint the tensor infos
    # (the header keeps its size because type and offset are fixed-width fields).
    T_F32_GGML, T_BF16_GGML = 0, 30
    promote = ("hc_attn_fn.weight", "hc_ffn_fn.weight", "indexer.attn_q_b.weight", "indexer.attn_k.weight",
               "indexer.proj.weight", "indexer_compressor_ape.weight", "indexer_compressor_gate.weight",
               "indexer.k_norm.weight", "indexer.k_norm.bias")
    size = os.path.getsize(args.model)
    appended = 0
    with open(args.model, "r+b") as f:
        for t in new_tensors:
            name, dims, typ, off = t
            local = name.split(".", 2)[-1] if name.startswith("blk.") else name
            if typ != T_BF16_GGML or local not in promote:
                continue
            n = math.prod(dims)
            f.seek(data_start + off)
            raw = np.frombuffer(f.read(2 * n), dtype="<u2").astype(np.uint32) << 16
            f32 = raw.view("<f4")
            new_off = (size - data_start + align - 1) // align * align
            f.seek(data_start + new_off)
            f.write(f32.astype("<f4").tobytes())
            size = data_start + new_off + 4 * n
            t[2], t[3] = T_F32_GGML, new_off
            appended += 1
    if appended:
        blob = fit_padding(version, new_kvs, new_tensors, data_start, align)
        print(f"promoted {appended} BF16 tensors to F32 (appended {(size - os.path.getsize(backup) * 0) / 1e6:.0f} MB at the end of the file)")
    with open(args.model, "r+b") as f:
        for off, n in a_log:
            f.seek(data_start + off)
            a = np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float64)
            if np.any(a <= -20) or np.any(a > 20):
                sys.exit("A_log values out of the expected range; refusing to convert twice")
            f.seek(data_start + off)
            f.write((-np.exp(a)).astype("<f4").tobytes())
        f.seek(0)
        f.write(blob)
    print(f"done; original header saved to {backup}")


if __name__ == "__main__":
    main()
