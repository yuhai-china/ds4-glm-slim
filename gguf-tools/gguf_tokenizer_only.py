#!/usr/bin/env python3
"""Extract the tokenizer metadata of a GGUF into a tiny tensor-less GGUF.

The GLM quantizers take ``--tokenizer-template FILE`` and read only the
``tokenizer.*`` key/value records of that file.  Keeping a 197 GiB model
around for that purpose is wasteful; this writes a GGUF v3 with zero tensors
that carries the same records byte for byte (plus ``general.architecture``
and ``general.name`` so the file is self-describing).

Usage:
  python3 gguf-tools/gguf_tokenizer_only.py IN.gguf OUT.gguf
  python3 gguf-tools/gguf_tokenizer_only.py IN.gguf OUT.gguf --verify

``--verify`` re-reads the output with the quantizer's loader and checks that
the records and the token list are identical to the input's.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glm53_quantize as q  # noqa: E402

KEEP_GENERAL = ("general.architecture", "general.name", "general.basename")


def read_kv_records(path):
    """Return (version, [(key, raw_record_bytes)]) for every metadata record."""
    records = []
    with open(path, "rb") as fp:
        if q.read_exact(fp, 4, "GGUF magic") != b"GGUF":
            q.fail(f"{path}: not a GGUF file")
        version = q.read_u32(fp, "GGUF version")
        if version not in (2, 3):
            q.fail(f"{path}: unsupported GGUF version {version}")
        q.read_u64(fp, "GGUF tensor count")
        n_kv = q.read_u64(fp, "GGUF metadata count")
        for _ in range(n_kv):
            start = fp.tell()
            key = q.read_gguf_string(fp, "GGUF metadata key")
            value_type = q.read_u32(fp, "GGUF metadata type")
            q.skip_gguf_value(fp, value_type)
            end = fp.tell()
            fp.seek(start)
            records.append((key, q.read_exact(fp, end - start, key)))
    return version, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="source GGUF (any DwarfStar or llama.cpp GGUF)")
    parser.add_argument("output", help="tokenizer-only GGUF to write")
    parser.add_argument("--name", help="general.name for the output (default: '<input name> tokenizer')")
    parser.add_argument("--verify", action="store_true", help="re-read the output and compare with the input")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        q.fail(f"output exists: {args.output}; use --overwrite")

    _version, records = read_kv_records(args.input)
    by_key = {key: raw for key, raw in records}
    kept = [(key, raw) for key, raw in records if key.startswith("tokenizer.")]
    if not any(key == "tokenizer.ggml.tokens" for key, _ in kept):
        q.fail(f"{args.input}: no tokenizer.ggml.tokens record")

    architecture = by_key.get("general.architecture")
    source_name = None
    if "general.name" in by_key:
        # decode the string value of general.name for the default output name
        raw = by_key["general.name"]
        key_len = struct.unpack("<Q", raw[:8])[0]
        value_type = struct.unpack("<I", raw[8 + key_len : 12 + key_len])[0]
        if value_type == q.GGUF_STRING:
            length = struct.unpack("<Q", raw[12 + key_len : 20 + key_len])[0]
            source_name = raw[20 + key_len : 20 + key_len + length].decode("utf-8", "replace")
    name = args.name or f"{source_name or os.path.basename(args.input)} tokenizer"

    out_records = []
    if architecture is not None:
        out_records.append(architecture)
    out_records.append(q.kv_string("general.name", name))
    out_records.append(q.kv_string("general.tokenizer_source", os.path.basename(args.input)))
    out_records.extend(raw for _, raw in kept)

    with open(args.output, "wb") as fp:
        fp.write(b"GGUF")
        fp.write(struct.pack("<IQQ", q.GGUF_VERSION, 0, len(out_records)))
        for raw in out_records:
            fp.write(raw)
        # data section is empty; pad the header to the GGUF alignment anyway
        fp.write(bytes(q.align(fp.tell()) - fp.tell()))
    size = os.path.getsize(args.output)
    print(f"gguf-tokenizer-only: wrote {args.output}: {len(kept)} tokenizer records, "
          f"{len(out_records)} records total, {size / (1 << 20):.1f} MiB", file=sys.stderr)

    if args.verify:
        in_records, in_tokens = q.load_tokenizer_records(args.input)
        out_records_read, out_tokens = q.load_tokenizer_records(args.output)
        if in_records != out_records_read:
            q.fail("verification failed: tokenizer records differ")
        if in_tokens != out_tokens:
            q.fail("verification failed: token lists differ")
        print(f"gguf-tokenizer-only: verified {len(out_tokens)} tokens, "
              f"{len(out_records_read)} records identical to {args.input}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"gguf-tokenizer-only: error: {error}", file=sys.stderr)
        sys.exit(1)
