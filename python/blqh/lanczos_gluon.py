"""Device-resident Lanczos for the Laughlin CSR.

CSR, the Krylov basis, and the working vector stay on the GPU. Each of the 40
steps launches Gluon matvec and reorthogonalization kernels and does not copy
the basis back. The steps-by-steps Rayleigh-Ritz matrix is copied once, at the
end, for a host eigh. No grid spin, no Ozaki, no async copy. Wave32 layouts;
memory ops are generic loads (the layer-2 anchor on gfx1201). buffer_load
offsets are not used, so nothing is pre-scaled by the element width.

Reorthogonalization is two-pass classical Gram-Schmidt: every dot against the
current w, then every axpy, then the same pair again. The first-pass dots are
the Rayleigh-Ritz column. For n <= 65536 one workgroup runs both passes, the
norm, and the normalize. Larger n keeps the original per-tile kernels, which
already move the basis at memory bandwidth.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from blqh.csrmv import _csrmv

MAX_STEPS = 40
MAT_BLOCK = 32
DOT_BLOCK = 128
DOT_WARPS = DOT_BLOCK // 32
# One workgroup eats a whole reorthogonalization segment. 4 wave32s; 8+ spilled
# and was several times slower on this GPU.
SEG_BLOCK = 128
SEG_WARPS = SEG_BLOCK // 32
# segment is one workgroup, so it is only the small-n path.
SEGMENT_N_MAX = 65536

# Published Lanczos E0 from bench/RESULTS.md (40 steps, full reorth).
REF_E0 = {
    (3, -1): -1.246507748,
    (4, -1): -1.672378787,
    (5, -1): -2.07620255,
    (6, -1): -2.485612885,
    (7, -1): -2.894718209,
    (7, 0): -2.894718209,
    (9, -1): -3.71900935,
}
DENSE_E0 = {
    (6, -1): -2.485612885,
    (7, 0): -2.894718209,
}


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _partial_dots(qr, qi, wr, wi, partial_r, partial_i, n, m, ld, BLOCK: gl.constexpr, STRIDE: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    idx = pid * BLOCK + offs
    mask = idx < n
    wrv = gl.load(wr + idx, mask=mask, other=0.0)
    wiv = gl.load(wi + idx, mask=mask, other=0.0)
    for k in range(m):
        base = k * ld
        qrv = gl.load(qr + base + idx, mask=mask, other=0.0)
        qiv = gl.load(qi + base + idx, mask=mask, other=0.0)
        # conj(q) * w
        dr = qrv * wrv + qiv * wiv
        di = qrv * wiv - qiv * wrv
        gl.store(partial_r + pid * STRIDE + k, gl.reduce(dr, 0, _add))
        gl.store(partial_i + pid * STRIDE + k, gl.reduce(di, 0, _add))


@gluon.jit
def _reduce_coefs(partial_r, partial_i, coef_r, coef_i, nblocks, store_k, kr, ki, kld, col,
                  BLOCK: gl.constexpr, STRIDE: gl.constexpr):
    k = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    acc_r = gl.zeros([BLOCK], gl.float64, layout=layout)
    acc_i = gl.zeros([BLOCK], gl.float64, layout=layout)
    for base in range(0, nblocks, BLOCK):
        idx = base + offs
        mask = idx < nblocks
        acc_r += gl.load(partial_r + idx * STRIDE + k, mask=mask, other=0.0)
        acc_i += gl.load(partial_i + idx * STRIDE + k, mask=mask, other=0.0)
    sr = gl.reduce(acc_r, 0, _add)
    si = gl.reduce(acc_i, 0, _add)
    gl.store(coef_r + k, sr)
    gl.store(coef_i + k, si)
    if store_k != 0:
        gl.store(kr + k * kld + col, sr)
        gl.store(ki + k * kld + col, si)


@gluon.jit
def _axpy_basis(qr, qi, wr, wi, cr, ci, partial_n, n, m, ld, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    idx = pid * BLOCK + offs
    mask = idx < n
    wrv = gl.load(wr + idx, mask=mask, other=0.0)
    wiv = gl.load(wi + idx, mask=mask, other=0.0)
    for k in range(m):
        dr = gl.load(cr + k)
        di = gl.load(ci + k)
        base = k * ld
        qrv = gl.load(qr + base + idx, mask=mask, other=0.0)
        qiv = gl.load(qi + base + idx, mask=mask, other=0.0)
        # w -= (dr + i di) * (qr + i qi)
        wrv -= dr * qrv - di * qiv
        wiv -= dr * qiv + di * qrv
    gl.store(wr + idx, wrv, mask=mask)
    gl.store(wi + idx, wiv, mask=mask)
    accn = wrv * wrv + wiv * wiv
    gl.store(partial_n + pid, gl.reduce(accn, 0, _add))


@gluon.jit
def _reduce_norm(partial, nblocks, betas, slot, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    acc = gl.zeros([BLOCK], gl.float64, layout=layout)
    for base in range(0, nblocks, BLOCK):
        idx = base + offs
        mask = idx < nblocks
        acc += gl.load(partial + idx, mask=mask, other=0.0)
    total = gl.reduce(acc, 0, _add)
    gl.store(betas + slot, gl.sqrt(total))


@gluon.jit
def _scale_store(wr, wi, qr, qi, n, betas, slot, BLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    idx = pid * BLOCK + offs
    mask = idx < n
    # A Python 1.0 is f32 and would round the new basis vector to ~1e-7.
    beta = gl.load(betas + slot)
    inv = gl.fdiv(gl.full([BLOCK], 1.0, gl.float64, layout=layout), beta)
    gl.store(qr + idx, gl.load(wr + idx, mask=mask, other=0.0) * inv, mask=mask)
    gl.store(qi + idx, gl.load(wi + idx, mask=mask, other=0.0) * inv, mask=mask)


def _cdiv(a, b):
    return (int(a) + int(b) - 1) // int(b)


@gluon.jit
def _full_dots(qr, qi, wr, wi, scratch_r, scratch_i, kr, ki, n, m, ld, kld, col, store_k,
               BLOCK: gl.constexpr, STRIDE: gl.constexpr):
    """One workgroup: q_k^H w for k < m, published as one scalar per coefficient."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    for k in range(m):
        acc_r = gl.zeros([BLOCK], gl.float64, layout=layout)
        acc_i = gl.zeros([BLOCK], gl.float64, layout=layout)
        for base in range(0, n, BLOCK):
            idx = base + offs
            mask = idx < n
            qrv = gl.load(qr + k * ld + idx, mask=mask, other=0.0)
            qiv = gl.load(qi + k * ld + idx, mask=mask, other=0.0)
            wrv = gl.load(wr + idx, mask=mask, other=0.0)
            wiv = gl.load(wi + idx, mask=mask, other=0.0)
            acc_r += qrv * wrv + qiv * wiv
            acc_i += qrv * wiv - qiv * wrv
        sr = gl.reduce(acc_r, 0, _add)
        si = gl.reduce(acc_i, 0, _add)
        gl.store(scratch_r + k, sr)
        gl.store(scratch_i + k, si)
        if store_k != 0:
            gl.store(kr + k * kld + col, sr)
            gl.store(ki + k * kld + col, si)


@gluon.jit
def _full_axpy(qr, qi, wr, wi, scratch_r, scratch_i, n, m, ld, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    accn = gl.zeros([BLOCK], gl.float64, layout=layout)
    for base in range(0, n, BLOCK):
        idx = base + offs
        mask = idx < n
        wrv = gl.load(wr + idx, mask=mask, other=0.0)
        wiv = gl.load(wi + idx, mask=mask, other=0.0)
        for k in range(m):
            dr = gl.load(scratch_r + k)
            di = gl.load(scratch_i + k)
            qrv = gl.load(qr + k * ld + idx, mask=mask, other=0.0)
            qiv = gl.load(qi + k * ld + idx, mask=mask, other=0.0)
            wrv = wrv - (dr * qrv - di * qiv)
            wiv = wiv - (dr * qiv + di * qrv)
        gl.store(wr + idx, wrv, mask=mask)
        gl.store(wi + idx, wiv, mask=mask)
        accn += wrv * wrv + wiv * wiv
    return gl.reduce(accn, 0, _add)


@gluon.jit
def _reorth_segment(qr, qi, wr, wi, scratch_r, scratch_i, kr, ki, betas, n, m, ld, kld, col, slot, do_axpy,
                    BLOCK: gl.constexpr, STRIDE: gl.constexpr):
    """Both CGS passes, the norm, and q_{j+1} = w / beta. One workgroup, no cross-block spin."""
    _full_dots(qr, qi, wr, wi, scratch_r, scratch_i, kr, ki, n, m, ld, kld, col, 1, BLOCK=BLOCK, STRIDE=STRIDE)
    if do_axpy == 0:
        return
    gl.barrier()
    _full_axpy(qr, qi, wr, wi, scratch_r, scratch_i, n, m, ld, BLOCK=BLOCK)
    _full_dots(qr, qi, wr, wi, scratch_r, scratch_i, kr, ki, n, m, ld, kld, col, 0, BLOCK=BLOCK, STRIDE=STRIDE)
    gl.barrier()
    nrm = _full_axpy(qr, qi, wr, wi, scratch_r, scratch_i, n, m, ld, BLOCK=BLOCK)
    beta = gl.sqrt(nrm)
    gl.store(betas + slot, beta)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    inv = gl.fdiv(gl.full([BLOCK], 1.0, gl.float64, layout=layout), beta)
    qbase = slot * ld
    for base in range(0, n, BLOCK):
        idx = base + offs
        mask = idx < n
        gl.store(qr + qbase + idx, gl.load(wr + idx, mask=mask, other=0.0) * inv, mask=mask)
        gl.store(qi + qbase + idx, gl.load(wi + idx, mask=mask, other=0.0) * inv, mask=mask)


def hermitian_from_upper(upper):
    """upper stores K[i, j] = q_i^H H q_j for i <= j; the rest is zero."""
    tri = np.triu(upper)
    full = tri + np.conjugate(np.triu(tri, 1)).T
    diag = np.real(np.diag(full))
    np.fill_diagonal(full, diag)
    return full


def load_csr(path):
    with np.load(path) as blob:
        row_ptr = np.array(blob["row_ptr"], dtype=np.int64)
        col = np.array(blob["col"], dtype=np.int32)
        val = np.array(blob["val"], dtype=np.complex128)
        meta = np.array(blob["meta"], dtype=np.int64)
    n = int(meta[3])
    if row_ptr.shape[0] != n + 1:
        raise RuntimeError(f"row_ptr length {row_ptr.shape[0]} != {n + 1}")
    return row_ptr, col, val, meta


def spot_rows(row_ptr, col, val, xr, xi, wr, wi, rows):
    worst = 0.0
    for i in rows:
        s = 0j
        for p in range(int(row_ptr[i]), int(row_ptr[i + 1])):
            c = int(col[p])
            s += val[p] * (xr[c] + 1j * xi[c])
        got = complex(float(wr[i]), float(wi[i]))
        denom = abs(s) if abs(s) > 0 else 1.0
        worst = max(worst, abs(got - s) / denom)
    return worst


class _Clock:
    def __init__(self):
        self.launches = 0
        self.pairs = []

    def begin(self):
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def end(self, token, bucket):
        token[1].record()
        self.pairs.append((token[0], token[1], bucket))

    def launch(self, bucket):
        self.launches += 1
        return self.begin()

    def totals(self):
        import torch

        torch.cuda.synchronize()
        acc = {"matvec": 0.0, "reorth": 0.0}
        for start, end, bucket in self.pairs:
            acc[bucket] += start.elapsed_time(end) * 1e-3
        return acc


class _Workspace:
    def __init__(self, row_ptr_d, col_d, vr, vi, steps):
        import torch

        self.row_ptr_d = row_ptr_d
        self.col_d = col_d
        self.vr = vr
        self.vi = vi
        self.steps = int(steps)
        self.n = int(row_ptr_d.shape[0]) - 1
        self.nblocks = _cdiv(self.n, DOT_BLOCK)
        dev = vr.device
        self.qr = torch.zeros(self.steps, self.n, dtype=torch.float64, device=dev)
        self.qi = torch.zeros(self.steps, self.n, dtype=torch.float64, device=dev)
        self.wr = torch.empty(self.n, dtype=torch.float64, device=dev)
        self.wi = torch.empty(self.n, dtype=torch.float64, device=dev)
        self.partial_r = torch.empty(self.nblocks * MAX_STEPS, dtype=torch.float64, device=dev)
        self.partial_i = torch.empty(self.nblocks * MAX_STEPS, dtype=torch.float64, device=dev)
        self.partial_n = torch.empty(self.nblocks, dtype=torch.float64, device=dev)
        self.scratch_r = torch.empty(MAX_STEPS, dtype=torch.float64, device=dev)
        self.scratch_i = torch.empty(MAX_STEPS, dtype=torch.float64, device=dev)
        self.coef_r = torch.empty(MAX_STEPS, dtype=torch.float64, device=dev)
        self.coef_i = torch.empty(MAX_STEPS, dtype=torch.float64, device=dev)
        self.kr = torch.zeros(self.steps, self.steps, dtype=torch.float64, device=dev)
        self.ki = torch.zeros(self.steps, self.steps, dtype=torch.float64, device=dev)
        self.betas = torch.zeros(self.steps, dtype=torch.float64, device=dev)

    def reset(self, q0r, q0i):
        import torch

        self.qr.zero_()
        self.qi.zero_()
        self.kr.zero_()
        self.ki.zero_()
        self.betas.zero_()
        self.qr[0].copy_(torch.from_numpy(np.ascontiguousarray(q0r)))
        self.qi[0].copy_(torch.from_numpy(np.ascontiguousarray(q0i)))


def _spot_first(ws, wr, wi, spot):
    import torch

    torch.cuda.synchronize()
    n = ws.n
    rows = spot["rows"]
    host_wr = np.zeros(n, dtype=np.float64)
    host_wi = np.zeros(n, dtype=np.float64)
    host_wr[rows] = wr[rows].detach().cpu().numpy()
    host_wi[rows] = wi[rows].detach().cpu().numpy()
    return spot_rows(spot["row_ptr"], spot["col"], spot["val"], spot["xr"], spot["xi"], host_wr, host_wi, rows)


def _matvec(ws, x_r, x_i, clock):
    token = clock.launch("matvec")
    _csrmv[(ws.n,)](
        ws.row_ptr_d, ws.col_d, ws.vr, ws.vi, x_r, x_i, ws.wr, ws.wi,
        BLOCK=MAT_BLOCK, num_warps=MAT_BLOCK // 32,
    )
    clock.end(token, "matvec")


def _lanczos_legacy(ws, steps, clock, spot=None):
    n = ws.n
    steps = int(steps)
    nblocks = ws.nblocks
    qr, qi = ws.qr, ws.qi
    wr, wi = ws.wr, ws.wi

    def dots(m, col, store_k):
        token = clock.launch("reorth")
        _partial_dots[(nblocks,)](
            qr, qi, wr, wi, ws.partial_r, ws.partial_i, n, m, n,
            BLOCK=DOT_BLOCK, STRIDE=MAX_STEPS, num_warps=DOT_WARPS,
        )
        clock.end(token, "reorth")
        token = clock.launch("reorth")
        _reduce_coefs[(m,)](
            ws.partial_r, ws.partial_i, ws.coef_r, ws.coef_i, nblocks, store_k,
            ws.kr, ws.ki, ws.steps, col, BLOCK=DOT_BLOCK, STRIDE=MAX_STEPS, num_warps=DOT_WARPS,
        )
        clock.end(token, "reorth")

    def axpy(m):
        token = clock.launch("reorth")
        _axpy_basis[(nblocks,)](
            qr, qi, wr, wi, ws.coef_r, ws.coef_i, ws.partial_n, n, m, n,
            BLOCK=DOT_BLOCK, num_warps=DOT_WARPS,
        )
        clock.end(token, "reorth")

    def normalize(slot):
        token = clock.launch("reorth")
        _reduce_norm[(1,)](ws.partial_n, nblocks, ws.betas, slot, BLOCK=DOT_BLOCK, num_warps=DOT_WARPS)
        clock.end(token, "reorth")
        token = clock.launch("reorth")
        _scale_store[(nblocks,)](
            wr, wi, qr[slot], qi[slot], n, ws.betas, slot, BLOCK=DOT_BLOCK, num_warps=DOT_WARPS
        )
        clock.end(token, "reorth")

    spot_err = None
    for j in range(steps):
        _matvec(ws, qr[j], qi[j], clock)
        if spot is not None and j == 0:
            spot_err = _spot_first(ws, wr, wi, spot)
        m = j + 1
        dots(m, j, 1)
        if j + 1 == steps:
            break
        axpy(m)
        dots(m, j, 0)
        axpy(m)
        normalize(j + 1)
    return spot_err


def _lanczos_segment(ws, steps, clock, spot=None):
    n = ws.n
    steps = int(steps)
    qr, qi = ws.qr, ws.qi
    wr, wi = ws.wr, ws.wi
    spot_err = None
    for j in range(steps):
        _matvec(ws, qr[j], qi[j], clock)
        if spot is not None and j == 0:
            spot_err = _spot_first(ws, wr, wi, spot)
        token = clock.launch("reorth")
        _reorth_segment[(1,)](
            qr, qi, wr, wi, ws.scratch_r, ws.scratch_i, ws.kr, ws.ki, ws.betas,
            n, j + 1, n, ws.steps, j, j + 1, 0 if j + 1 == steps else 1,
            BLOCK=SEG_BLOCK, STRIDE=MAX_STEPS, num_warps=SEG_WARPS,
        )
        clock.end(token, "reorth")
    return spot_err


def resolve_fuse(mode, n):
    if mode == "auto":
        return "segment" if int(n) <= SEGMENT_N_MAX else "legacy"
    if mode not in ("segment", "legacy"):
        raise ValueError(f"unknown fuse mode {mode}")
    return mode


def lanczos(ws, steps, clock, spot=None, fuse="auto"):
    """Forty (or fewer) steps. Buffers in `ws` are already on device."""
    kind = resolve_fuse(fuse, ws.n)
    if kind == "legacy":
        return _lanczos_legacy(ws, steps, clock, spot)
    return _lanczos_segment(ws, steps, clock, spot)


def ritz(upper):
    herm = hermitian_from_upper(upper)
    evals, evecs = np.linalg.eigh(herm, UPLO="L")
    return herm, evals, evecs


def residual(ws, coeff, energy):
    import torch

    cr = torch.tensor(np.ascontiguousarray(coeff.real), dtype=torch.float64, device=ws.vr.device)
    ci = torch.tensor(np.ascontiguousarray(coeff.imag), dtype=torch.float64, device=ws.vr.device)
    psi_r = cr @ ws.qr - ci @ ws.qi
    psi_i = cr @ ws.qi + ci @ ws.qr
    nrm = torch.sqrt((psi_r * psi_r + psi_i * psi_i).sum())
    psi_r = psi_r / nrm
    psi_i = psi_i / nrm
    n = ws.n
    yr = torch.empty(n, dtype=torch.float64, device=ws.vr.device)
    yi = torch.empty(n, dtype=torch.float64, device=ws.vr.device)
    _csrmv[(n,)](ws.row_ptr_d, ws.col_d, ws.vr, ws.vi, psi_r, psi_i, yr, yi, BLOCK=MAT_BLOCK, num_warps=MAT_BLOCK // 32)
    torch.cuda.synchronize()
    dr = yr - energy * psi_r
    di = yi - energy * psi_i
    res = torch.sqrt((dr * dr + di * di).sum())
    return float(res.detach().cpu()) / max(abs(energy), 1e-30)


def start_vector(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    q /= np.linalg.norm(q)
    return np.ascontiguousarray(q.real), np.ascontiguousarray(q.imag)


def upload(row_ptr, col, val):
    import torch

    dev = "cuda"
    row_ptr_d = torch.tensor(row_ptr, dtype=torch.int64, device=dev)
    col_d = torch.tensor(col, dtype=torch.int32, device=dev)
    vr = torch.tensor(np.ascontiguousarray(val.real), dtype=torch.float64, device=dev)
    vi = torch.tensor(np.ascontiguousarray(val.imag), dtype=torch.float64, device=dev)
    return row_ptr_d, col_d, vr, vi


def basis_norm_err(ws, steps):
    import torch

    err = 0.0
    for j in range(int(steps)):
        nrm = torch.sum(ws.qr[j] * ws.qr[j] + ws.qi[j] * ws.qi[j])
        err = max(err, abs(float(nrm.detach().cpu()) - 1.0))
    return err


def run_case(path, steps, seed, do_residual, fuse):
    import torch

    row_ptr, col, val, meta = load_csr(path)
    n = int(meta[3])
    steps = min(int(steps), n, MAX_STEPS)
    q0r, q0i = start_vector(n, seed)
    row_ptr_d, col_d, vr, vi = upload(row_ptr, col, val)
    torch.cuda.synchronize()
    rows = [0, min(n // 2, n - 1), n - 1]
    spot = {"row_ptr": row_ptr, "col": col, "val": val, "xr": q0r, "xi": q0i, "rows": rows}
    kind = resolve_fuse(fuse, n)
    ws = _Workspace(row_ptr_d, col_d, vr, vi, steps)
    # Compile every kernel (matvec plus the fused reorth path, including the last step).
    t_compile = time.perf_counter()
    warm_steps = 2 if steps >= 2 else 1
    ws.reset(q0r, q0i)
    spot_err = lanczos(ws, warm_steps, _Clock(), spot=spot, fuse=kind)
    torch.cuda.synchronize()
    t_compile = time.perf_counter() - t_compile
    if spot_err is not None and spot_err > 1e-12:
        raise RuntimeError(f"matvec spot check relative error {spot_err:.3e}")

    ws.reset(q0r, q0i)
    torch.cuda.synchronize()
    clock = _Clock()
    t0 = time.perf_counter()
    lanczos(ws, steps, clock, spot=None, fuse=kind)
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0
    gpu_times = clock.totals()
    t1 = time.perf_counter()
    upper = torch.complex(ws.kr, ws.ki).detach().cpu().numpy()
    beta = ws.betas.detach().cpu().numpy()
    herm, evals, evecs = ritz(upper)
    t_ritz = time.perf_counter() - t1
    e0 = float(evals[0])
    diag_imag = float(np.max(np.abs(np.imag(np.diag(upper)))))
    res = None
    if do_residual:
        res = residual(ws, evecs[:, 0], e0)
    beta_min = float(np.min(beta[1:steps])) if steps > 1 else float("nan")
    torch.cuda.synchronize()
    bnorm = basis_norm_err(ws, steps)
    return {
        "n": n,
        "nnz": int(meta[4]),
        "steps": steps,
        "E0": e0,
        "diag_imag": diag_imag,
        "beta_min": beta_min,
        "launches": clock.launches,
        "t_gpu": t_gpu,
        "t_matvec": gpu_times["matvec"],
        "t_reorth": gpu_times["reorth"],
        "t_ritz": t_ritz,
        "t_compile": t_compile,
        "spot": spot_err,
        "residual": res,
        "herm_asym": float(np.max(np.abs(herm - herm.conj().T))),
        "fuse": kind,
        "basis_norm_err": bnorm,
    }


def rel_err(got, ref):
    return abs(got - ref) / max(abs(ref), 1e-30)


def ground_state(row_ptr, col, val, q0, steps):
    """One Lanczos run. `q0` is the normalized complex start vector.

    Returns `(energy, ritz_values, psi)`. At most `MAX_STEPS` steps. n <= 65536
    uses the one-workgroup reorthogonalization; larger n uses the tiled kernels.
    """
    import torch

    steps = int(steps)
    n = int(row_ptr.shape[0]) - 1
    if steps < 1:
        raise ValueError("steps must be positive")
    if steps > MAX_STEPS:
        raise ValueError(f"gluon lanczos supports at most {MAX_STEPS} steps")
    steps = min(steps, n)
    q0 = np.asarray(q0, dtype=np.complex128)
    q0r = np.ascontiguousarray(q0.real)
    q0i = np.ascontiguousarray(q0.imag)
    row_ptr_d, col_d, vr, vi = upload(row_ptr, col, val)
    kind = resolve_fuse("auto", n)
    ws = _Workspace(row_ptr_d, col_d, vr, vi, steps)
    warm = 2 if steps >= 2 else 1
    ws.reset(q0r, q0i)
    lanczos(ws, warm, _Clock(), fuse=kind)
    torch.cuda.synchronize()
    ws.reset(q0r, q0i)
    lanczos(ws, steps, _Clock(), fuse=kind)
    torch.cuda.synchronize()
    upper = torch.complex(ws.kr, ws.ki).detach().cpu().numpy()
    _, evals, evecs = ritz(upper)
    coeff = evecs[:, 0]
    cr = torch.tensor(np.ascontiguousarray(coeff.real), dtype=torch.float64, device=vr.device)
    ci = torch.tensor(np.ascontiguousarray(coeff.imag), dtype=torch.float64, device=vr.device)
    psi_r = cr @ ws.qr - ci @ ws.qi
    psi_i = cr @ ws.qi + ci @ ws.qr
    nrm = torch.sqrt((psi_r * psi_r + psi_i * psi_i).sum())
    psi = ((psi_r + 1j * psi_i) / nrm).detach().cpu().numpy()
    return float(evals[0]), np.asarray(evals[:steps], dtype=np.float64), np.ascontiguousarray(psi)


def _read_bin(path):
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != b"BLQHGL01":
            raise RuntimeError(f"bad lanczos request magic {magic!r}")
        n, steps = np.fromfile(fh, dtype=np.int32, count=2)
        nnz = int(np.fromfile(fh, dtype=np.int64, count=1)[0])
        row_ptr = np.fromfile(fh, dtype=np.int64, count=int(n) + 1)
        col = np.fromfile(fh, dtype=np.int32, count=nnz)
        val = np.fromfile(fh, dtype=np.complex128, count=nnz)
        q0 = np.fromfile(fh, dtype=np.complex128, count=int(n))
    return row_ptr, col, val, q0, int(steps)


def _write_result(path, energy, evals, psi):
    evals = np.asarray(evals, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.complex128)
    with open(path, "wb") as fh:
        fh.write(b"BLQHGO01")
        np.array([psi.shape[0], evals.shape[0]], dtype=np.int32).tofile(fh)
        np.array([energy], dtype=np.float64).tofile(fh)
        evals.tofile(fh)
        psi.tofile(fh)


def solve_bin(request, result):
    import torch

    torch.cuda.set_device(0)
    row_ptr, col, val, q0, steps = _read_bin(request)
    energy, evals, psi = ground_state(row_ptr, col, val, q0, steps)
    _write_result(result, energy, evals, psi)


def main():
    parser = argparse.ArgumentParser(description="Fused Gluon Lanczos on one dumped CSR")
    parser.add_argument("--bin", type=Path, help="C++ request written by lanczos_gluon.cpp")
    parser.add_argument("--result", type=Path, help="where --bin writes energy and psi")
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--nel", type=int)
    parser.add_argument("--J", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--no-residual", action="store_false", dest="residual")
    parser.add_argument("--fuse", choices=("auto", "segment", "legacy"), default="auto")
    args = parser.parse_args()
    if args.bin is not None or args.result is not None:
        if args.bin is None or args.result is None:
            raise SystemExit("--bin and --result are used together")
        solve_bin(args.bin, args.result)
        return 0
    if args.npz is None or args.nel is None or args.J is None:
        raise SystemExit("need --npz --nel --J, or --bin and --result")
    held = __import__("os").environ.get("GEAK_GPU_LOCK_HELD")
    if held != "1":
        raise SystemExit(f"refusing to run without gpu_lock on GPU 1 (GEAK_GPU_LOCK_HELD={held})")
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected one visible GPU, found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    out = run_case(args.npz, args.steps, args.seed, args.residual, args.fuse)
    ref = REF_E0.get((args.nel, args.J))
    dense = DENSE_E0.get((args.nel, args.J))
    rel = None if ref is None else rel_err(out["E0"], ref)
    rel_dense = None if dense is None else rel_err(out["E0"], dense)
    wall = out["t_gpu"] + out["t_ritz"]
    print(
        "BLQH_GL_LANCZOS"
        f" nel={args.nel} J={args.J} n={out['n']} nnz={out['nnz']} steps={out['steps']}"
        f" E0={out['E0']:.12g} rel={rel} rel_dense={rel_dense}"
        f" wall={wall:.6g} t_gpu={out['t_gpu']:.6g} t_matvec={out['t_matvec']:.6g}"
        f" t_reorth={out['t_reorth']:.6g} t_ritz={out['t_ritz']:.6g}"
        f" launches={out['launches']} compile={out['t_compile']:.6g}"
        f" residual={out['residual']} spot={out['spot']}"
        f" diag_imag={out['diag_imag']:.3e} beta_min={out['beta_min']:.3e}"
        f" herm_asym={out['herm_asym']:.3e} fuse={out['fuse']}"
        f" basis_norm_err={out['basis_norm_err']:.3e}",
        flush=True,
    )
    if out["spot"] is not None and out["spot"] > 1e-12:
        return 2
    if not np.isfinite(out["basis_norm_err"]) or out["basis_norm_err"] > 1e-6:
        return 4
    # The 1e-8 gate is for RESULTS rows that were themselves 40 steps.
    # N=8/N=9 in that table used 80 steps, so a 40-step rerun is not that gate.
    forty_step_ref = {(3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (7, 0)}
    if (
        rel is not None
        and (args.nel, args.J) in forty_step_ref
        and rel > 1e-8
        and (rel_dense is None or rel_dense > 1e-8)
    ):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
