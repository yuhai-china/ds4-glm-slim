#!/usr/bin/env python3
"""In-place header fix for DwarfStar GLM GGUFs: store `<arch>.context_length` as u32
(llama.cpp's loader requires u32) and keep the data offset unchanged by padding the
header with a `general.padding` string.  Usage: gguf_fix_ctx_u32.py MODEL.gguf"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5next_to_llamacpp import read_header, fit_padding, T_U32, T_U64  # noqa: E402


def main():
    path = sys.argv[1]
    version, kvs, tensors, header_end, data_start, align = read_header(path)
    changed = 0
    for e in kvs:
        if e[0].endswith(".context_length") and e[1] == T_U64 and e[2] < 2**32:
            e[1] = T_U32
            changed += 1
    kvs = [e for e in kvs if e[0] != "general.padding"]
    if not changed:
        print(f"{path}: nothing to do")
        return
    blob = fit_padding(version, kvs, tensors, data_start, align)
    with open(path, "rb") as f:
        orig = f.read(data_start)
    with open(path + ".header.bak", "wb") as f:
        f.write(orig)
    with open(path, "r+b") as f:
        f.write(blob)
    print(f"{path}: context_length -> u32, header rewritten in place ({len(blob)} B, data offset {data_start} unchanged); "
          f"backup {path}.header.bak")


if __name__ == "__main__":
    main()
