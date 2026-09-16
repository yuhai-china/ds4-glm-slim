#!/usr/bin/env python3
"""Scalar float32 reference of ds4q_write_iq2_xxs_block for debugging.

Every operation is performed on np.float32 scalars in the same order as the
C code, so the result should be byte-identical to libds4quants.  It is far
too slow for real conversions; it exists to localize discrepancies between
the C quantizer and the CUDA port stage by stage.
"""

from __future__ import annotations

import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iq2xxs_cuda  # noqa: E402

F = np.float32
GRID, GMAP, NEIGHBOURS, COUNTS = iq2xxs_cuda.build_tables()


def nearest_int(v):
    return int(np.rint(F(v)))  # round half to even, like the 12582912 trick


def make_qp_quants(x, qw, nmax=4):
    n = len(x)
    mx = F(0)
    for v in x:
        mx = max(mx, v)
    if mx < F(1e-15):
        return F(0), [0] * n
    iscale = F(F(nmax) / mx)
    L = [nearest_int(iscale * x[i]) & 0xFF for i in range(n)]
    scale = F(F(1) / iscale)
    best_mse = F(0)
    for i in range(n):
        diff = F(x[i] - F(scale * F(L[i])))
        best_mse = F(best_mse + F(F(qw[i] * diff) * diff))
    for is_ in range(-4, 5):
        if is_ == 0:
            continue
        iscale_is = F(F(F(F(0.1) * F(is_)) + F(nmax)) / mx)
        scale_is = F(F(1) / iscale_is)
        mse = F(0)
        for i in range(n):
            l = min(nmax, nearest_int(iscale_is * x[i]))
            diff = F(x[i] - F(scale_is * F(l)))
            mse = F(mse + F(F(qw[i] * diff) * diff))
        if mse < best_mse:
            best_mse = mse
            iscale = iscale_is
    sumlx = F(0)
    suml2 = F(0)
    for i in range(n):
        l = min(nmax, nearest_int(iscale * x[i]))
        L[i] = l & 0xFF
        w = qw[i]
        sumlx = F(sumlx + F(F(w * x[i]) * F(l)))
        suml2 = F(suml2 + F(F(w * F(l)) * F(l)))
    for _ in range(5):
        changed = 0
        for i in range(n):
            w = qw[i]
            li = F(L[i])
            slx = F(sumlx - F(F(w * x[i]) * li))
            sl2 = F(suml2 - F(F(w * li) * li))
            if slx > 0 and sl2 > 0:
                new_l = min(nmax, nearest_int(F(F(x[i] * sl2) / slx)))
                if new_l != L[i]:
                    slx = F(slx + F(F(w * x[i]) * F(new_l)))
                    sl2 = F(sl2 + F(F(w * F(new_l)) * F(new_l)))
                    if F(F(slx * slx) * suml2) > F(F(sumlx * sumlx) * sl2):
                        L[i] = new_l & 0xFF
                        sumlx = slx
                        suml2 = sl2
                        changed += 1
        if not changed:
            break
    return (F(sumlx / suml2) if suml2 > 0 else F(0)), L


def find_best_neighbour(u, xval, waux, scale):
    best_d2 = F(np.finfo(np.float32).max)
    best = -1
    for j in range(COUNTS[u]):
        gi = int(NEIGHBOURS[u, j])
        d2 = F(0)
        for i in range(8):
            q = F(GRID[gi, i])
            diff = F(F(scale * q) - xval[i])
            d2 = F(d2 + F(F(waux[i] * diff) * diff))
        if d2 < best_d2:
            best_d2 = d2
            best = gi
    return best


def code_of(L8):
    u = 0
    for i in range(8):
        u |= L8[i] << (2 * i)
    return u


def project(u, xval, waux, scale):
    gi = GMAP[u]
    if gi < 0:
        gi = find_best_neighbour(u, xval, waux, scale)
    return int(gi), [int((GRID[gi, i] - 1) // 2) for i in range(8)]


def write_block(x, qw, trace=None):
    """x, qw: 256 float32 values. Returns 66 bytes."""
    x = [F(v) for v in x]
    qw = [F(v) for v in qw]
    k_max_q = 3
    q2 = [0] * 16
    scales = [F(0)] * 8
    sumx2 = F(0)
    for v in x:
        sumx2 = F(sumx2 + F(v * v))
    sigma2 = F(sumx2 / F(256))
    max_scale = F(0)
    for ib in range(8):
        xb = x[32 * ib : 32 * ib + 32]
        qwb = qw[32 * ib : 32 * ib + 32]
        weight = [F(qwb[i] * F(np.sqrt(F(sigma2 + F(xb[i] * xb[i]))))) for i in range(32)]
        waux = [F(np.sqrt(w)) for w in weight]
        xval = [F(0)] * 32
        block_signs = [0] * 4
        for k in range(4):
            nflip = 0
            s = 0
            for i in range(8):
                v = xb[8 * k + i]
                if v >= 0:
                    xval[8 * k + i] = v
                else:
                    xval[8 * k + i] = F(-v)
                    nflip += 1
                    s |= 1 << i
            if nflip % 2:
                imin = 0
                mn = F(F(weight[8 * k] * xb[8 * k]) * xb[8 * k])
                for i in range(1, 8):
                    ax = F(F(weight[8 * k + i] * xb[8 * k + i]) * xb[8 * k + i])
                    if ax < mn:
                        mn = ax
                        imin = i
                xval[8 * k + imin] = F(-xval[8 * k + imin])
                s ^= 1 << imin
            block_signs[k] = s & 127
        mx = xval[0]
        for v in xval[1:]:
            mx = max(mx, v)
        if mx < F(1e-15):
            scales[ib] = F(0)
            continue
        scale, _ = make_qp_quants(xval, weight, k_max_q + 1)
        eff_max = F(scale * F(k_max_q))
        if trace is not None:
            trace.setdefault("qp_scale", []).append(float(scale))
        if eff_max <= 0:
            scales[ib] = F(0)
            continue
        best = F(0)
        L = [0] * 32
        chosen_is = None
        for is_ in range(-6, 7):
            id_ = F(F(F(2 * k_max_q - 1) + F(F(is_) * F(0.1))) / eff_max)
            this_scale = F(F(1) / id_)
            Laux = [0] * 32
            for k in range(4):
                l8 = []
                for i in range(8):
                    l = nearest_int(F(F(0.5) * F(F(id_ * xval[8 * k + i]) - F(1))))
                    l8.append(max(0, min(k_max_q - 1, l)))
                _, l8 = project(code_of(l8), xval[8 * k : 8 * k + 8], waux[8 * k : 8 * k + 8], this_scale)
                Laux[8 * k : 8 * k + 8] = l8
            sumqx = F(0)
            sumq2 = F(0)
            for i in range(32):
                w = weight[i]
                q = F(2 * Laux[i] + 1)
                sumqx = F(sumqx + F(F(w * xval[i]) * q))
                sumq2 = F(sumq2 + F(F(w * q) * q))
            if sumq2 > 0 and F(sumqx * sumqx) > F(best * sumq2):
                scale = F(sumqx / sumq2)
                best = F(scale * sumqx)
                L = list(Laux)
                chosen_is = is_
        if trace is not None:
            trace.setdefault("loop_scale", []).append(float(scale))
            trace.setdefault("chosen_is", []).append(chosen_is)
        if scale > 0:
            id_ = F(F(1) / scale)
            for k in range(4):
                l8 = []
                for i in range(8):
                    l = nearest_int(F(F(0.5) * F(F(id_ * xval[8 * k + i]) - F(1))))
                    l8.append(max(0, min(k_max_q - 1, l)))
                _, l8 = project(code_of(l8), xval[8 * k : 8 * k + 8], waux[8 * k : 8 * k + 8], scale)
                L[8 * k : 8 * k + 8] = l8
            sumqx = F(0)
            sumq2 = F(0)
            for i in range(32):
                w = weight[i]
                q = F(2 * L[i] + 1)
                sumqx = F(sumqx + F(F(w * xval[i]) * q))
                sumq2 = F(sumq2 + F(F(w * q) * q))
            if sumq2 > 0:
                scale = F(sumqx / sumq2)
        if scale < 0:
            scale = F(-scale)
            block_signs = [(~s) & 127 for s in block_signs]
        for k in range(4):
            gi = int(GMAP[code_of(L[8 * k : 8 * k + 8])])
            assert gi >= 0
            q2[2 * ib] |= gi << (8 * k)
            q2[2 * ib + 1] |= block_signs[k] << (7 * k)
        scales[ib] = scale
        max_scale = max(max_scale, scale)
        if trace is not None:
            trace.setdefault("final_scale", []).append(float(scale))
    if max_scale == 0:
        return struct.pack("<e", 0.0) + bytes(64)
    d = F(max_scale / F(31))
    id_ = F(F(1) / d)
    for ib in range(8):
        l = nearest_int(F(F(0.5) * F(F(id_ * scales[ib]) - F(1))))
        q2[2 * ib + 1] |= max(0, min(15, l)) << 28
    return struct.pack("<e", float(d)) + struct.pack("<16I", *q2)
