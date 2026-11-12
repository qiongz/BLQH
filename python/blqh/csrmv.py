"""One-shot Gluon CSR matvec. Kept only if N=4 or N=5 matches the CPU product
and is not slower. This is not a tuning loop."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _csrmv(row_ptr, col, vr, vi, xr, xi, yr, yi, BLOCK: gl.constexpr):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [BLOCK // 32], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    start = gl.load(row_ptr + row)
    end = gl.load(row_ptr + row + 1)
    acc_r = gl.zeros([BLOCK], gl.float64, layout=layout)
    acc_i = gl.zeros([BLOCK], gl.float64, layout=layout)
    nnz = end - start
    for base in range(0, nnz, BLOCK):
        p = start + base + offs
        mask = p < end
        c = gl.load(col + p, mask=mask, other=0).to(gl.int64)
        ar = gl.load(vr + p, mask=mask, other=0.0)
        ai = gl.load(vi + p, mask=mask, other=0.0)
        xre = gl.load(xr + c, mask=mask, other=0.0)
        xim = gl.load(xi + c, mask=mask, other=0.0)
        acc_r += ar * xre - ai * xim
        acc_i += ar * xim + ai * xre
    yr_p = yr + row
    yi_p = yi + row
    gl.store(yr_p, gl.reduce(acc_r, 0, _add))
    gl.store(yi_p, gl.reduce(acc_i, 0, _add))


def csrmv(row_ptr, col, val, x, block=32):
    import torch

    n = int(x.shape[0])
    if block % 32 != 0:
        raise ValueError("block must be a multiple of the wave size")
    yr = torch.zeros(n, dtype=torch.float64, device=x.device)
    yi = torch.zeros(n, dtype=torch.float64, device=x.device)
    vr = val.real.contiguous()
    vi = val.imag.contiguous()
    xr = x.real.contiguous()
    xi = x.imag.contiguous()
    _csrmv[(n,)](row_ptr, col, vr, vi, xr, xi, yr, yi, BLOCK=block, num_warps=block // 32)
    return torch.complex(yr, yi)
