#!/usr/bin/env python3
"""IQ2_XXS quantization on a CUDA GPU with PyTorch.

This is a line-by-line port of ``ds4q_write_iq2_xxs_block`` in ``quants.c``
(itself the GGML IQ2_XXS search) that processes every 32-value block of a
whole tensor batch in parallel.  Grid, direct map and two-shell neighbour
lists are built with the same rules as the C initializer, float32 sums are
accumulated in the same order, rounding is round-half-to-even like
``ds4q_nearest_int`` and the uint8 wrap-around of ``make_qp_quants`` is kept,
so for identical inputs and importance weights the output bytes match the C
quantizer.  ``tests/test_iq2xxs_cuda.py`` checks that against libds4quants.

The point is speed: one 2048x6144 expert matrix costs about 7 s per CPU core
in C and a few tens of milliseconds here, which turns a full GLM-5.3-class
routed-expert conversion from many hours into minutes.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - reported by the caller
    torch = None

QK_K = 256
BLOCK = 32
K_MAX_Q = 3
QP_NMAX = K_MAX_Q + 1
GROUP_MAX_EPS = 1e-15
IQ2_XXS_BLOCK_BYTES = 66
MAP_SIZE = 43692
NEIGHBOUR_SHELLS = 2

# Same table as ds4q_iq2_xxs_init / GGML kgrid_2bit_256.
KGRID = [
        0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
     1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
     1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
     2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
     4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
     5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
     8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
]


def f32(value):
    """Round a Python number to float32 and return it as a Python float.

    The C code mixes float literals and int promotions in float32; scalar
    constants are pre-rounded so that tensor ops see exactly the same values.
    """
    return float(np.float32(value))



def sdiv(num, t):
    """Correctly rounded float32 ``num / t`` for a Python scalar numerator.

    ``num / tensor`` in PyTorch is ``tensor.reciprocal() * num`` (two
    roundings); the C code performs one IEEE division.
    """
    return torch.full_like(t, num) / t


def tdiv(t, num):
    """Correctly rounded float32 ``t / num`` for a Python scalar divisor.

    PyTorch's CUDA ``tensor / scalar`` multiplies by the reciprocal instead
    ("may lose one bit of precision"); tensor / tensor is a true division.
    """
    return t / torch.full_like(t, num)

def build_tables():
    """Grid values, direct map and padded two-shell neighbour lists (NumPy)."""
    kgrid = np.array(KGRID, dtype=np.int64)
    grid = np.zeros((256, 8), dtype=np.int64)
    for i in range(8):
        grid[:, i] = 2 * ((kgrid >> (2 * i)) & 3) + 1

    gmap = np.full(MAP_SIZE, -1, dtype=np.int64)
    gmap[kgrid] = np.arange(256, dtype=np.int64)

    codes = np.arange(MAP_SIZE, dtype=np.int64)
    pos = np.stack([2 * ((codes >> (2 * k)) & 3) + 1 for k in range(8)], axis=1)
    d2 = ((pos[:, None, :] - grid[None, :, :]) ** 2).sum(axis=2)  # [MAP_SIZE, 256]
    # qsort on (d2, j): ascending distance, ties by grid index.
    order = np.argsort(d2, axis=1, kind="stable")
    d2s = np.take_along_axis(d2, order, axis=1)
    # Keep the first NEIGHBOUR_SHELLS distinct distance values.
    shell_change = np.concatenate(
        [np.zeros((MAP_SIZE, 1), dtype=bool), d2s[:, 1:] > d2s[:, :-1]], axis=1
    )
    shell_index = np.cumsum(shell_change, axis=1)  # 0 for first shell, 1 second...
    keep = shell_index < NEIGHBOUR_SHELLS
    counts = keep.sum(axis=1)
    max_n = int(counts.max())
    neighbours = np.where(keep, order, -1)[:, :max_n].astype(np.int64)
    return grid, gmap, neighbours, counts



NEAREST_GRID_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

// One thread per group of eight values.  Mirrors ds4q_iq2_find_best_neighbour:
// direct map hit -> that grid index, otherwise the first neighbour (list
// order) with the smallest sqrt(weight)-weighted distance at the given scale.
// Compiled with -fmad=false so every mul/add rounds like the C code.
__global__ void nearest_grid_kernel(
        const int32_t* __restrict__ u,
        const float* __restrict__ xval,   // [8, N] coordinate-major
        const float* __restrict__ waux,   // [8, N]
        const float* __restrict__ scale,  // [N]
        const int32_t* __restrict__ gmap, // [65536]
        const int32_t* __restrict__ nb_count, // [65536]
        const int16_t* __restrict__ nb_index, // [65536, max_n]
        const uint64_t* __restrict__ grid_packed, // [256]
        int64_t* __restrict__ out,        // [N]
        int64_t N,
        int max_n) {
    int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;
    int code = (int)u[g];
    int hit = gmap[code];
    if (hit >= 0) { out[g] = hit; return; }
    float xv[8], wa[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        xv[i] = xval[(int64_t)i * N + g];
        wa[i] = waux[(int64_t)i * N + g];
    }
    const float sc = scale[g];
    const int n = nb_count[code];
    const int16_t* nb = nb_index + (int64_t)code * max_n;
    float best_d2 = FLT_MAX;
    int best = -1;
    for (int j = 0; j < n; ++j) {
        const int gi = nb[j];
        const uint64_t gp = grid_packed[gi];
        float d2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float q = (float)((int)((gp >> (8 * i)) & 0xFFu));
            const float diff = sc * q - xv[i];
            const float t = wa[i] * diff;
            d2 = d2 + t * diff;
        }
        if (d2 < best_d2) { best_d2 = d2; best = gi; }
    }
    out[g] = best;
}

torch::Tensor nearest_grid(torch::Tensor u, torch::Tensor xval, torch::Tensor waux,
                           torch::Tensor scale, torch::Tensor gmap, torch::Tensor nb_count,
                           torch::Tensor nb_index, torch::Tensor grid_packed) {
    TORCH_CHECK(u.is_cuda() && u.dtype() == torch::kInt32 && u.is_contiguous());
    TORCH_CHECK(xval.dtype() == torch::kFloat32 && xval.is_contiguous() && xval.dim() == 2 && xval.size(0) == 8);
    TORCH_CHECK(waux.dtype() == torch::kFloat32 && waux.is_contiguous() && waux.sizes() == xval.sizes());
    TORCH_CHECK(scale.dtype() == torch::kFloat32 && scale.is_contiguous());
    TORCH_CHECK(gmap.dtype() == torch::kInt32 && nb_count.dtype() == torch::kInt32);
    TORCH_CHECK(nb_index.dtype() == torch::kInt16 && nb_index.is_contiguous() && nb_index.dim() == 2);
    TORCH_CHECK(grid_packed.dtype() == torch::kInt64 && grid_packed.numel() == 256);
    const int64_t N = u.numel();
    TORCH_CHECK(xval.size(1) == N && scale.numel() == N);
    auto out = torch::empty({N}, u.options().dtype(torch::kInt64));
    if (N == 0) return out;
    const int threads = 256;
    const int64_t blocks = (N + threads - 1) / threads;
    nearest_grid_kernel<<<(unsigned)blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        u.data_ptr<int32_t>(), xval.data_ptr<float>(), waux.data_ptr<float>(), scale.data_ptr<float>(),
        gmap.data_ptr<int32_t>(), nb_count.data_ptr<int32_t>(), nb_index.data_ptr<int16_t>(),
        reinterpret_cast<const uint64_t*>(grid_packed.data_ptr<int64_t>()),
        out.data_ptr<int64_t>(), N, (int)nb_index.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

NEAREST_GRID_CPP = "torch::Tensor nearest_grid(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"

_KERNEL_MODULE = None
_KERNEL_ERROR = None


def _find_cuda_home():
    """Prefer a toolkit matching the torch CUDA major (pip nvidia/cu13 etc.)."""
    import glob
    import os

    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return None
    major = (torch.version.cuda or "").split(".")[0]
    if not major:
        return None
    for path in sorted(glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "nvidia", f"cu{major}"))):
        if os.path.isfile(os.path.join(path, "bin", "nvcc")):
            return os.path.abspath(path)
    system = f"/usr/local/cuda-{major}"
    if os.path.isfile(os.path.join(system, "bin", "nvcc")):
        return system
    return None


def load_nearest_grid_kernel(verbose=False):
    """Compile (once, cached by torch) the fused neighbour-search kernel.

    Returns the extension module or None when nvcc/toolchain is unavailable;
    callers fall back to the pure PyTorch search, which yields the same bytes.
    """
    global _KERNEL_MODULE, _KERNEL_ERROR
    if _KERNEL_MODULE is not None or _KERNEL_ERROR is not None:
        return _KERNEL_MODULE
    import os
    import sys

    try:
        home = _find_cuda_home()
        if home:
            os.environ["CUDA_HOME"] = home
        from torch.utils.cpp_extension import load_inline

        _KERNEL_MODULE = load_inline(
            name="ds4_iq2xxs_nearest_grid",
            cpp_sources=[NEAREST_GRID_CPP],
            cuda_sources=[NEAREST_GRID_CUDA],
            functions=["nearest_grid"],
            extra_cuda_cflags=["-O3", "-fmad=false", "--prec-div=true", "--prec-sqrt=true", "-ftz=false"],
            verbose=verbose,
        )
    except Exception as error:  # noqa: BLE001 - any build failure means fallback
        _KERNEL_ERROR = error
        print(f"iq2xxs_cuda: fused neighbour kernel unavailable ({type(error).__name__}: {str(error).splitlines()[0][:200]}); using PyTorch search", file=sys.stderr)
    return _KERNEL_MODULE


class IQ2XXSCudaQuantizer:
    # Off-grid groups are searched in buckets of neighbour-list width so that
    # padding stays small (the usage-weighted median list has ~37 entries,
    # the longest 150).
    BUCKETS = (8, 16, 24, 32, 40, 48, 64, 80, 100, 150)

    def __init__(self, device="cuda", use_kernel=True):
        if torch is None:
            raise RuntimeError("PyTorch is required for the CUDA IQ2_XXS quantizer")
        self.device = torch.device(device)
        dev = self.device
        grid, gmap, neighbours, counts = build_tables()
        max_n = neighbours.shape[1]
        self.grid_q = torch.tensor(grid, dtype=torch.float32, device=dev)  # [256, 8]
        self.grid_l = torch.tensor((grid - 1) // 2, dtype=torch.int8, device=dev)
        # Index with any 16-bit pattern; codes >= MAP_SIZE never occur for l <= 2.
        gmap_full = np.full(65536, -1, dtype=np.int64)
        gmap_full[:MAP_SIZE] = gmap
        self.gmap = torch.tensor(gmap_full, dtype=torch.int64, device=dev)
        nb_full = np.full((65536, max_n), -1, dtype=np.int64)
        nb_full[:MAP_SIZE] = neighbours
        counts_full = np.zeros(65536, dtype=np.int64)
        counts_full[:MAP_SIZE] = counts
        self.nb_count = torch.tensor(counts_full, dtype=torch.int64, device=dev)
        self.nb_index = torch.tensor(nb_full, dtype=torch.int16, device=dev)  # [65536, max_n]
        # Grid coordinate i of neighbour j of code u, as int8: [65536, 8, max_n].
        nb_q = np.zeros((65536, 8, max_n), dtype=np.int8)
        valid = nb_full >= 0
        for i in range(8):
            coord = grid[:, i][np.where(valid, nb_full, 0)]
            nb_q[:, i, :] = np.where(valid, coord, 0)
        self.nb_q = torch.tensor(nb_q, dtype=torch.int8, device=dev)
        self.max_n = max_n
        # Kernel-side tables.
        self.gmap32 = self.gmap.to(torch.int32)
        self.nb_count32 = self.nb_count.to(torch.int32)
        packed = np.zeros(256, dtype=np.uint64)
        for i in range(8):
            packed |= grid[:, i].astype(np.uint64) << np.uint64(8 * i)
        self.grid_packed = torch.tensor(packed.view(np.int64), dtype=torch.int64, device=dev)
        self.kernel = load_nearest_grid_kernel() if use_kernel else None
        self.shifts2 = 2 * torch.arange(8, device=dev, dtype=torch.int64)
        self.shifts1 = torch.arange(8, device=dev, dtype=torch.int64)
        self.shifts8 = 8 * torch.arange(4, device=dev, dtype=torch.int64)
        self.shifts7 = 7 * torch.arange(4, device=dev, dtype=torch.int64)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _seq_sum(terms):
        """Sequential float32 accumulation in C order: acc = 0; acc += t_i."""
        acc = torch.zeros_like(terms[0])
        for term in terms:
            acc = acc + term
        return acc

    def _pack_u(self, l):
        """l: [..., 8] int64 in 0..3 -> 16-bit code with l_i at bits 2i."""
        return (l << self.shifts2).sum(dim=-1)

    def _nearest_grid(self, u, xval_g, waux_g, scale):
        """Direct map or two-shell neighbour search for groups of eight.

        u: [N] codes, xval_g/waux_g: [8, N] (coordinate-major), scale: [N].
        Returns grid index [N].
        """
        if self.kernel is not None:
            return self.kernel.nearest_grid(
                u.contiguous(), xval_g, waux_g, scale.contiguous(),
                self.gmap32, self.nb_count32, self.nb_index, self.grid_packed,
            )
        u = u.to(torch.int64)
        gidx = self.gmap[u]
        off = gidx < 0
        if not bool(off.any()):
            return gidx
        gidx = gidx.clone()
        idx_all = off.nonzero(as_tuple=True)[0]
        cnt = self.nb_count[u[idx_all]]
        order = torch.argsort(cnt)
        idx_sorted = idx_all[order]
        cnt_sorted = cnt[order]
        lo = 0
        for width in self.BUCKETS:
            hi = int(torch.searchsorted(cnt_sorted, torch.tensor([width], device=self.device), right=True).item())
            if hi <= lo:
                continue
            idx = idx_sorted[lo:hi]
            uu = u[idx]
            sc = scale[idx][:, None]
            d2 = torch.zeros((idx.numel(), width), dtype=torch.float32, device=self.device)
            for i in range(8):
                qi = self.nb_q[:, i, :width][uu].to(torch.float32)
                diff = sc * qi - xval_g[i][idx][:, None]
                d2 = d2 + (waux_g[i][idx][:, None] * diff) * diff
            valid = torch.arange(width, device=self.device)[None, :] < cnt_sorted[lo:hi][:, None]
            d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
            best = d2.argmin(dim=1)  # first minimum, like the strict < in C
            chosen = self.nb_index[:, :width][uu].gather(1, best[:, None])[:, 0].to(torch.int64)
            gidx[idx] = chosen
            lo = hi
            del d2, valid, qi, diff
        return gidx

    def _make_qp_quants(self, x, qw):
        """Port of ds4q_make_qp_quants(32, 4, xval, L, weight).

        x, qw: [32, G] (element-major).  Returns scale [G].
        """
        nmax = QP_NMAX
        mx = torch.clamp(x.max(dim=0).values, min=0.0)
        ok = mx >= GROUP_MAX_EPS
        mx_safe = torch.where(ok, mx, torch.ones_like(mx))
        iscale = sdiv(f32(nmax), mx_safe)
        l0 = torch.round(iscale[None, :] * x)
        lu8 = (l0.to(torch.int64) & 0xFF).to(torch.float32)
        scale = sdiv(1.0, iscale)
        diff = x - scale[None, :] * lu8
        best_mse = self._seq_sum([(qw[i] * diff[i]) * diff[i] for i in range(BLOCK)])
        for is_ in range(-4, 5):
            if is_ == 0:
                continue
            num = f32(np.float32(0.1) * np.float32(is_) + np.float32(nmax))
            iscale_is = sdiv(num, mx_safe)
            scale_is = sdiv(1.0, iscale_is)
            l = torch.clamp(torch.round(iscale_is[None, :] * x), max=float(nmax))
            diff = x - scale_is[None, :] * l
            mse = self._seq_sum([(qw[i] * diff[i]) * diff[i] for i in range(BLOCK)])
            upd = mse < best_mse
            best_mse = torch.where(upd, mse, best_mse)
            iscale = torch.where(upd, iscale_is, iscale)
        l = torch.clamp(torch.round(iscale[None, :] * x), max=float(nmax))
        lu8 = (l.to(torch.int64) & 0xFF).to(torch.float32)
        sumlx = self._seq_sum([(qw[i] * x[i]) * l[i] for i in range(BLOCK)])
        suml2 = self._seq_sum([(qw[i] * l[i]) * l[i] for i in range(BLOCK)])
        cols = [lu8[i].clone() for i in range(BLOCK)]
        for _try in range(5):
            for i in range(BLOCK):
                w = qw[i]
                xi = x[i]
                li = cols[i]
                slx = sumlx - (w * xi) * li
                sl2 = suml2 - (w * li) * li
                cond = (slx > 0) & (sl2 > 0)
                slx_safe = torch.where(cond, slx, torch.ones_like(slx))
                new_l = torch.clamp(torch.round((xi * sl2) / slx_safe), max=float(nmax))
                changed = cond & (new_l != li)
                slx2 = slx + (w * xi) * new_l
                sl22 = sl2 + (w * new_l) * new_l
                better = changed & ((slx2 * slx2) * suml2 > (sumlx * sumlx) * sl22)
                new_u8 = (torch.where(better, new_l, li).to(torch.int64) & 0xFF).to(torch.float32)
                cols[i] = torch.where(better, new_u8, li)
                sumlx = torch.where(better, slx2, sumlx)
                suml2 = torch.where(better, sl22, suml2)
        safe = torch.where(suml2 > 0, suml2, torch.ones_like(suml2))
        scale = torch.where(suml2 > 0, sumlx / safe, torch.zeros_like(suml2))
        return torch.where(ok, scale, torch.zeros_like(scale))

    def _round_l(self, id_, xval):
        """l = clamp(nearest_int(0.5f * (id * xval - 1)), 0, K_MAX_Q - 1); [32, G] -> [32, G]."""
        l = torch.round(0.5 * (id_[None, :] * xval - 1.0))
        return torch.clamp(l, min=0.0, max=float(K_MAX_Q - 1)).to(torch.int8)

    def _sums(self, weight, xval, l):
        """sumqx / sumq2 over the 32 elements; all inputs [32, G]."""
        q = (2 * l.to(torch.int16) + 1).to(torch.float32)
        sumqx = self._seq_sum([(weight[i] * xval[i]) * q[i] for i in range(BLOCK)])
        sumq2 = self._seq_sum([(weight[i] * q[i]) * q[i] for i in range(BLOCK)])
        return sumqx, sumq2

    def _search(self, lT, xval_c, waux_c, scale, G):
        """Round-trip through the grid: lT [32, G] -> grid index per group [G*4]."""
        # element index e = 8k + i; group-major code needs l_i at bits 2i.
        l_g = lT.reshape(4, 8, G)  # [k, i, G]
        u = l_g[:, 0, :].to(torch.int32)
        for i in range(1, 8):
            u = u | (l_g[:, i, :].to(torch.int32) << (2 * i))  # [4, G]
        u = u.t().reshape(G * 4).contiguous()  # group index g*4 + k
        return self._nearest_grid(u, xval_c, waux_c, scale.repeat_interleave(4))

    # -- main entry --------------------------------------------------------

    @torch.no_grad()
    def encode(self, x, qw):
        """Quantize x [R, C] float32 (CUDA) with importance weights qw.

        qw is either [C] (one importance value per column, shared by all
        rows) or [R, C] (per-row weights, used to batch several matrices with
        their own imatrix slices into one call).  Returns a uint8 CUDA tensor
        of R * (C / 256) * 66 bytes in GGUF row order.  C must be a multiple
        of 256.
        """
        if x.dim() != 2 or x.shape[1] % QK_K:
            raise ValueError(f"IQ2_XXS needs [rows, cols] with cols % 256 == 0, got {tuple(x.shape)}")
        dev = self.device
        x = x.to(dev, torch.float32).contiguous()
        qw = qw.to(dev, torch.float32)
        R, C = x.shape
        S = R * (C // QK_K)
        xs = x.reshape(S, QK_K)
        if qw.dim() == 1:
            if qw.numel() != C:
                raise ValueError(f"imatrix width {qw.numel()} does not match tensor width {C}")
            qws = qw.reshape(1, C // QK_K, QK_K).expand(R, -1, -1).reshape(S, QK_K)
        else:
            if tuple(qw.shape) != (R, C):
                raise ValueError(f"per-row weights {tuple(qw.shape)} do not match tensor {tuple(x.shape)}")
            qws = qw.contiguous().reshape(S, QK_K)

        # Element-major copies: xsT[i] is element i of every superblock.
        xsT = xs.t().contiguous()  # [256, S]
        sumx2 = self._seq_sum([xsT[i] * xsT[i] for i in range(QK_K)])
        sigma2 = tdiv(sumx2, f32(QK_K))
        del xsT

        G = S * (QK_K // BLOCK)
        xb = xs.reshape(G, BLOCK)
        qwb = qws.reshape(G, BLOCK)
        sig = sigma2.repeat_interleave(QK_K // BLOCK)
        weight = qwb * torch.sqrt(sig[:, None] + xb * xb)
        waux = torch.sqrt(weight)

        # Signs with even parity per group of eight.
        neg = xb < 0
        xval = torch.where(neg, -xb, xb)
        neg_g = neg.reshape(G, 4, 8)
        s = (neg_g.to(torch.int64) << self.shifts1).sum(dim=2)  # [G, 4]
        nflip = neg_g.sum(dim=2)
        odd = (nflip % 2) == 1
        ax = ((weight * xb) * xb).reshape(G, 4, 8)
        imin = ax.argmin(dim=2)  # [G, 4] first minimum
        del ax, neg
        xval_g = xval.reshape(G, 4, 8)
        flip = torch.zeros_like(xval_g, dtype=torch.bool)
        flip.scatter_(2, imin[:, :, None], odd[:, :, None])
        xval_g = torch.where(flip, -xval_g, xval_g)
        s = torch.where(odd, s ^ (1 << imin), s)
        block_signs = s & 127
        del flip, s, nflip, imin, odd

        xval = xval_g.reshape(G, BLOCK)
        # Coordinate-major views for the neighbour search: [8, G*4].
        xval_c = xval_g.reshape(G * 4, 8).t().contiguous()
        waux_c = waux.reshape(G * 4, 8).t().contiguous()
        # Element-major for the sequential loops: [32, G].
        xvalT = xval.t().contiguous()
        weightT = weight.t().contiguous()
        del xb, qwb, xval_g, waux, weight

        mx = xvalT.max(dim=0).values
        valid = mx >= GROUP_MAX_EPS
        scale = self._make_qp_quants(xvalT, weightT)
        eff_max = scale * f32(K_MAX_Q)
        valid = valid & (eff_max > 0)
        eff_safe = torch.where(valid, eff_max, torch.ones_like(eff_max))

        LT = torch.zeros((BLOCK, G), dtype=torch.int8, device=dev)
        best = torch.zeros(G, dtype=torch.float32, device=dev)
        for is_ in range(-6, 7):
            num = f32(np.float32(2 * K_MAX_Q - 1) + np.float32(is_) * np.float32(0.1))
            id_ = sdiv(num, eff_safe)
            this_scale = sdiv(1.0, id_)
            laux = self._round_l(id_, xvalT)
            gidx = self._search(laux, xval_c, waux_c, this_scale, G)
            laux = self.grid_l[gidx].reshape(G, BLOCK).t().contiguous()  # [32, G]
            sumqx, sumq2 = self._sums(weightT, xvalT, laux)
            upd = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            new_scale = sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2))
            scale = torch.where(upd, new_scale, scale)
            best = torch.where(upd, scale * sumqx, best)
            LT = torch.where(upd[None, :], laux, LT)

        pos = scale > 0
        pos_safe = torch.where(pos, scale, torch.ones_like(scale))
        id_ = sdiv(1.0, pos_safe)
        l2 = self._round_l(id_, xvalT)
        gidx = self._search(l2, xval_c, waux_c, pos_safe, G)
        l2 = self.grid_l[gidx].reshape(G, BLOCK).t().contiguous()
        LT = torch.where(pos[None, :], l2, LT)
        sumqx, sumq2 = self._sums(weightT, xvalT, LT)
        refine = pos & (sumq2 > 0)
        scale = torch.where(refine, sumqx / torch.where(sumq2 > 0, sumq2, torch.ones_like(sumq2)), scale)
        del xval_c, waux_c, xvalT, weightT, laux, l2

        negs = scale < 0
        scale = torch.where(negs, -scale, scale)
        block_signs = torch.where(negs[:, None], (~block_signs) & 127, block_signs)

        # Pack grid indices and signs; invalid blocks stay all-zero.
        L = LT.t().reshape(G, 4, 8).to(torch.int64)
        u = self._pack_u(L)  # [G, 4]
        gidx = self.gmap[u]
        if bool(((gidx < 0) & valid[:, None]).any()):
            # Only reachable through the degenerate scale <= 0 path; project.
            gidx = torch.where(gidx < 0, torch.zeros_like(gidx), gidx)
        gidx = torch.where(valid[:, None], gidx, torch.zeros_like(gidx))
        block_signs = torch.where(valid[:, None], block_signs, torch.zeros_like(block_signs))
        scale = torch.where(valid, scale, torch.zeros_like(scale))
        q2a = (gidx << self.shifts8).sum(dim=1)  # [G]
        q2b = (block_signs << self.shifts7).sum(dim=1)

        scales = scale.reshape(S, QK_K // BLOCK)
        max_scale = scales.max(dim=1).values
        nz = max_scale > 0
        d = tdiv(max_scale, f32(31))
        d_safe = torch.where(nz, d, torch.ones_like(d))
        id_d = sdiv(1.0, d_safe)
        ls = torch.clamp(torch.round(0.5 * (id_d[:, None] * scales - 1.0)), min=0.0, max=15.0).to(torch.int64)
        q2b = q2b.reshape(S, QK_K // BLOCK) | (ls << 28)
        q2a = q2a.reshape(S, QK_K // BLOCK)
        q2a = torch.where(nz[:, None], q2a, torch.zeros_like(q2a))
        q2b = torch.where(nz[:, None], q2b, torch.zeros_like(q2b))
        d16 = torch.where(nz, d, torch.zeros_like(d)).to(torch.float16)

        words = torch.stack([q2a, q2b], dim=2).reshape(S, 16)
        # uint32 bit patterns -> int32 storage (two's complement, no saturation)
        words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
        out = torch.empty((S, IQ2_XXS_BLOCK_BYTES), dtype=torch.uint8, device=dev)
        out[:, 0:2] = d16.view(torch.uint8).reshape(S, 2)
        out[:, 2:] = words.view(torch.uint8).reshape(S, 64)
        return out.reshape(-1)


def fp8_e4m3_lut(np_module=np):
    """Float32 value of every FP8 E4M3 code, identical to glm53_quantize."""
    import math

    values = np_module.zeros(256, dtype=np_module.float32)
    for code in range(256):
        absolute = code & 0x7F
        if absolute == 0x7F:
            value = math.nan
        else:
            exponent = (code >> 3) & 0x0F
            mantissa = code & 0x07
            value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
            if code & 0x80:
                value = -value
        values[code] = value
    return values


class CudaExpertEncoder:
    """FP8 block-128 safetensors payload -> IQ2_XXS bytes, all on the GPU."""

    def __init__(self, device="cuda", use_kernel=True):
        self.quantizer = IQ2XXSCudaQuantizer(device, use_kernel=use_kernel)
        self.device = self.quantizer.device
        self.lut = torch.tensor(fp8_e4m3_lut(), dtype=torch.float32, device=self.device)

    def dequantize_fp8(self, codes_u8, scales_f32, shape):
        """codes: bytes/ndarray of shape, scales: [ceil(r/128), ceil(c/128)] f32."""
        rows, cols = shape
        codes = torch.frombuffer(bytearray(codes_u8), dtype=torch.uint8).to(self.device).reshape(rows, cols)
        scales = torch.frombuffer(bytearray(scales_f32), dtype=torch.float32).to(self.device)
        scales = scales.reshape((rows + 127) // 128, (cols + 127) // 128)
        expanded = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
        return self.lut[codes.to(torch.int64)] * expanded

    def encode(self, x, qw=None):
        """x: [rows, cols] float32 CUDA tensor; qw: [cols] weights or None.

        Without weights the same weight-energy fallback as the CPU quantizer
        is used: importance[column] = sum(row[column]^2).
        """
        if qw is None:
            qw = torch.sum(x * x, dim=0)
        else:
            qw = torch.as_tensor(np.ascontiguousarray(qw, dtype=np.float32)).to(self.device)
        return self.quantizer.encode(x, qw)

    def encode_to_bytes(self, x, qw=None):
        return self.encode(x, qw).cpu().numpy().tobytes()

    def encode_batch(self, matrices, weights):
        """Quantize several equally shaped matrices in one call.

        matrices: list of [rows, cols] float32 CUDA tensors; weights: list of
        [cols] importance vectors (NumPy/torch) or None per matrix.  Returns
        one bytes object per matrix.
        """
        if not matrices:
            return []
        rows, cols = matrices[0].shape
        x = torch.cat(matrices, dim=0)
        qw_rows = []
        for m, w in zip(matrices, weights):
            if w is None:
                w = torch.sum(m * m, dim=0)
            else:
                w = torch.as_tensor(np.ascontiguousarray(w, dtype=np.float32)).to(self.device)
            qw_rows.append(w.reshape(1, cols).expand(rows, cols))
        qw = torch.cat(qw_rows, dim=0)
        out = self.quantizer.encode(x, qw).cpu().numpy()
        per = out.size // len(matrices)
        return [out[i * per : (i + 1) * per].tobytes() for i in range(len(matrices))]
