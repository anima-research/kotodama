"""Fused energy routing: kotodama.model.energy_route.route_energy_split in five Triton kernels (CUDA, bf16 sources).

Same op as the PyTorch path (``--attn_res_energy_impl torch``); every big tensor is read once per kernel:
- F-A ``_terms_kernel``: per source and token, ρ = rsqrt(mean(x²) + ε) and [P | ℓ] = ρ · x [nw ⊙ U | qw] (one tl.dot).
- F-B ``_steps_mix_kernel``: the mirror-descent steps per token in registers, then h = Σ α_s x_s.
- B-1 ``_g_kernel``: g_s = dh · x_s. B-S ``_steps_bwd_kernel``: the steps' closed-form backward per token (dℓ, dP, dσ,
  the dρ coefficient). B-2 ``_dx_kernel``: dx for every source with split-K partials of dW = Σ ρ x ⊗ dP and
  dq = Σ ρ dℓ x.
Saved for backward: ρ, [P | ℓ] and α per site (small fp32), never the per-step tensors. Registered as
``kotodama::energy_split_fwd`` / ``energy_split_bwd`` custom ops, so compiled regions treat them as opaque.
``fused_supported`` says when a call can run here; ``route_energy_split_fused`` falls back to the PyTorch path otherwise.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    AVAILABLE = True
except ImportError:  # CPU-only installs: the PyTorch path serves every call
    AVAILABLE = False

if AVAILABLE:

    @triton.jit
    def _terms_kernel(src_ptr, wq_ptr, rho_ptr, pl_ptr, s_off, BT, eps,
                      D: tl.constexpr, R_: tl.constexpr, RW_: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        """F-A: for source (s_off + program_id(1)) and BLOCK_T tokens: ρ, and [P | ℓ] = ρ · x [W | q] (fp32)."""
        pid_t = tl.program_id(0)
        s = tl.program_id(1)
        t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        dd = tl.arange(0, BLOCK_D)
        jj = tl.arange(0, RW_)
        src = src_ptr + s.to(tl.int64) * BT * D
        acc = tl.zeros((BLOCK_T, RW_), dtype=tl.float32)
        ss = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            x = tl.load(src + t[:, None].to(tl.int64) * D + d0 + dd[None, :])
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
            w = tl.load(wq_ptr + (d0 + dd)[:, None] * RW_ + jj[None, :])
            acc = tl.dot(x, w, acc)
        rho = tl.rsqrt(ss / D + eps)
        so = (s_off + s).to(tl.int64)
        tl.store(rho_ptr + so * BT + t, rho)
        tl.store(pl_ptr + (so * BT + t[:, None].to(tl.int64)) * RW_ + jj[None, :], acc * rho[:, None])


    @triton.jit
    def _src(committed_ptr, partial_ptr, s: tl.constexpr, N_COMMITTED: tl.constexpr, BT, D: tl.constexpr):
        if s < N_COMMITTED:
            return committed_ptr + s * BT * D
        else:
            return partial_ptr


    @triton.jit
    def _row(m, sid, s: tl.constexpr):
        return tl.sum(tl.where(sid[:, None] == s, m, 0.0), axis=0)


    @triton.jit
    def _softmax0(z):
        mx = tl.max(z, axis=0)
        e = tl.exp(z - mx[None, :])
        return e / tl.sum(e, axis=0)[None, :]


    @triton.jit
    def _steps_fwd(pl_ptr, sig, ell, t, BT, sid,
                   S: tl.constexpr, R_: tl.constexpr, RW_: tl.constexpr, K: tl.constexpr):
        """a^K from ℓ (S_PAD, BLOCK_T; -inf on padding), P (loaded per source), σ (R,)."""
        rr = tl.arange(0, R_)
        a = _softmax0(ell)
        for _k in tl.static_range(K):
            c = tl.zeros((t.shape[0], R_), dtype=tl.float32)
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                c += _row(a, sid, s)[:, None] * Ps
            sc = c * sig[None, :]
            z = ell
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                z = tl.where(sid[:, None] == s, z + tl.sum(Ps * sc, axis=1)[None, :], z)
            a = _softmax0(z)
        return a


    @triton.jit
    def _steps_mix_kernel(committed_ptr, partial_ptr, pl_ptr, sig_ptr, a_ptr, h_ptr, BT,
                          D: tl.constexpr, N_COMMITTED: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr,
                          R_: tl.constexpr, RW_: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr,
                          BLOCK_D: tl.constexpr):
        """F-B: the steps per token, store α (S, BT), then h = Σ α_s x_s."""
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        sid = tl.arange(0, S_PAD)
        rr = tl.arange(0, R_)
        sig = tl.load(sig_ptr + rr).to(tl.float32)
        ell = tl.load(pl_ptr + (sid[:, None] * BT + t[None, :].to(tl.int64)) * RW_ + R_, mask=sid[:, None] < S,
                      other=float("-inf"))
        a = _steps_fwd(pl_ptr, sig, ell, t, BT, sid, S, R_, RW_, K)
        tl.store(a_ptr + sid[:, None] * BT + t[None, :], a, mask=sid[:, None] < S)
        dd = tl.arange(0, BLOCK_D)
        for d0 in range(0, D, BLOCK_D):
            h = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for s in tl.static_range(S):
                src = _src(committed_ptr, partial_ptr, s, N_COMMITTED, BT, D)
                x = tl.load(src + t[:, None].to(tl.int64) * D + d0 + dd[None, :]).to(tl.float32)
                h += _row(a, sid, s)[:, None] * x
            tl.store(h_ptr + t[:, None].to(tl.int64) * D + d0 + dd[None, :], h.to(h_ptr.dtype.element_ty))


    @triton.jit
    def _g_kernel(committed_ptr, partial_ptr, dh_ptr, g_ptr, BT,
                  D: tl.constexpr, N_COMMITTED: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr,
                  BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        """B-1: g_s = dh · x_s per token."""
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        sid = tl.arange(0, S_PAD)
        dd = tl.arange(0, BLOCK_D)
        g = tl.zeros((S_PAD, BLOCK_T), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            dh = tl.load(dh_ptr + t[:, None].to(tl.int64) * D + d0 + dd[None, :]).to(tl.float32)
            for s in tl.static_range(S):
                src = _src(committed_ptr, partial_ptr, s, N_COMMITTED, BT, D)
                x = tl.load(src + t[:, None].to(tl.int64) * D + d0 + dd[None, :]).to(tl.float32)
                g = tl.where(sid[:, None] == s, g + tl.sum(dh * x, axis=1)[None, :], g)
        tl.store(g_ptr + sid[:, None] * BT + t[None, :], g, mask=sid[:, None] < S)


    @triton.jit
    def _steps_bwd_kernel(pl_ptr, rho_ptr, sig_ptr, g_ptr, dl_ptr, dp_ptr, coef_ptr, dsig_part_ptr, BT,
                          D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, R_: tl.constexpr, RW_: tl.constexpr,
                          K: tl.constexpr, BLOCK_T: tl.constexpr):
        """B-S: backward of the K steps per token (forward recomputed in registers). Writes dℓ (S, BT), dP (S, BT, R),
        coef_s = dρ_s ρ_s³ / D (S, BT), and this program's dσ partial (R,)."""
        pid = tl.program_id(0)
        t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        sid = tl.arange(0, S_PAD)
        rr = tl.arange(0, R_)
        smask = sid[:, None] < S
        sig = tl.load(sig_ptr + rr).to(tl.float32)
        ell = tl.load(pl_ptr + (sid[:, None] * BT + t[None, :].to(tl.int64)) * RW_ + R_, mask=smask, other=float("-inf"))
        # forward recompute, keeping a^0..a^K (K <= 3 supported by the static unroll)
        a0 = _softmax0(ell)
        a_list_0 = a0
        a_prev = a0
        a1 = a0
        a2 = a0
        a3 = a0
        for k in tl.static_range(K):
            c = tl.zeros((BLOCK_T, R_), dtype=tl.float32)
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                c += _row(a_prev, sid, s)[:, None] * Ps
            sc = c * sig[None, :]
            z = ell
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                z = tl.where(sid[:, None] == s, z + tl.sum(Ps * sc, axis=1)[None, :], z)
            a_new = _softmax0(z)
            if k == 0:
                a1 = a_new
            elif k == 1:
                a2 = a_new
            else:
                a3 = a_new
            a_prev = a_new
        # zero dP (it accumulates per step)
        for s in tl.static_range(S):
            tl.store(dp_ptr + (s * BT + t[:, None].to(tl.int64)) * R_ + rr[None, :], tl.zeros((BLOCK_T, R_), tl.float32))
        g = tl.load(g_ptr + sid[:, None] * BT + t[None, :], mask=smask, other=0.0)
        da = g
        dl = tl.zeros((S_PAD, BLOCK_T), dtype=tl.float32)
        dsig = tl.zeros((R_,), dtype=tl.float32)
        for kk in tl.static_range(K):
            k = K - 1 - kk                       # step k maps a^k -> a^(k+1)
            if k == 0:
                a_in = a_list_0
                a_out = a1
            elif k == 1:
                a_in = a1
                a_out = a2
            else:
                a_in = a2
                a_out = a3
            dz = a_out * (da - tl.sum(a_out * da, axis=0)[None, :])
            dz = tl.where(smask, dz, 0.0)
            dl += dz
            c = tl.zeros((BLOCK_T, R_), dtype=tl.float32)
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                c += _row(a_in, sid, s)[:, None] * Ps
            sc = c * sig[None, :]
            dc = tl.zeros((BLOCK_T, R_), dtype=tl.float32)
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                dz_s = _row(dz, sid, s)
                dc += dz_s[:, None] * Ps * sig[None, :]
                dsig += tl.sum(dz_s[:, None] * Ps * c, axis=0)
                dp_addr = dp_ptr + (s * BT + t[:, None].to(tl.int64)) * R_ + rr[None, :]
                tl.store(dp_addr, tl.load(dp_addr) + dz_s[:, None] * sc)
            da_new = tl.zeros((S_PAD, BLOCK_T), dtype=tl.float32)
            for s in tl.static_range(S):
                Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
                da_new = tl.where(sid[:, None] == s, tl.sum(Ps * dc, axis=1)[None, :], da_new)
                dp_addr = dp_ptr + (s * BT + t[:, None].to(tl.int64)) * R_ + rr[None, :]
                tl.store(dp_addr, tl.load(dp_addr) + _row(a_in, sid, s)[:, None] * dc)
            da = da_new
        dz0 = a_list_0 * (da - tl.sum(a_list_0 * da, axis=0)[None, :])
        dl += tl.where(smask, dz0, 0.0)
        tl.store(dl_ptr + sid[:, None] * BT + t[None, :], dl, mask=smask)
        rho = tl.load(rho_ptr + sid[:, None] * BT + t[None, :], mask=smask, other=1.0)
        coef = tl.zeros((S_PAD, BLOCK_T), dtype=tl.float32)
        for s in tl.static_range(S):
            Ps = tl.load(pl_ptr + (s * BT + t[:, None].to(tl.int64)) * RW_ + rr[None, :])
            dPs = tl.load(dp_ptr + (s * BT + t[:, None].to(tl.int64)) * R_ + rr[None, :])
            rho_s = _row(rho, sid, s)
            ell_s = _row(tl.where(smask, ell, 0.0), sid, s)
            drho = (_row(dl, sid, s) * ell_s + tl.sum(dPs * Ps, axis=1)) / rho_s
            coef = tl.where(sid[:, None] == s, (drho * rho_s * rho_s * rho_s / D)[None, :], coef)
        tl.store(coef_ptr + sid[:, None] * BT + t[None, :], coef, mask=smask)
        tl.store(dsig_part_ptr + pid * R_ + rr, dsig)


    @triton.jit
    def _dx_one(x_base, dx_base, s, dh, t, d, rr, W, q, a_ptr, dl_ptr, dp_ptr, coef_ptr, rho_ptr, dw, dq, BT,
                D: tl.constexpr, R_: tl.constexpr):
        """One source's dx tile and its contribution to the dW / dq partials (runtime s: one set of dot buffers)."""
        x = tl.load(x_base + t[:, None].to(tl.int64) * D + d[None, :])
        xf = x.to(tl.float32)
        so = (BT * s).to(tl.int64)          # BT is a runtime int; s may be specialised to a constexpr (n == 1)
        a_s = tl.load(a_ptr + so + t)
        dl_s = tl.load(dl_ptr + so + t)
        co_s = tl.load(coef_ptr + so + t)
        rho_s = tl.load(rho_ptr + so + t)
        dP = tl.load(dp_ptr + (so + t[:, None].to(tl.int64)) * R_ + rr[None, :])                     # (BLOCK_T, R) fp32
        wdp = tl.dot(dP.to(tl.bfloat16), tl.trans(W))                                                 # (BLOCK_T, BLOCK_D)
        dx = a_s[:, None] * dh + rho_s[:, None] * (dl_s[:, None] * q[None, :] + wdp) - co_s[:, None] * xf
        tl.store(dx_base + t[:, None].to(tl.int64) * D + d[None, :], dx.to(dx_base.dtype.element_ty))
        dw = tl.dot(tl.trans(x), (dP * rho_s[:, None]).to(tl.bfloat16), dw)
        dq += tl.sum(xf * (rho_s * dl_s)[:, None], axis=0)
        return dw, dq


    @triton.jit
    def _dx_kernel(committed_ptr, partial_ptr, dh_ptr, a_ptr, dl_ptr, dp_ptr, coef_ptr, rho_ptr, w_ptr, q_ptr,
                   dcommitted_ptr, dpartial_ptr, dw_part_ptr, dq_part_ptr, BT, T_CHUNK, N_COMMITTED,
                   D: tl.constexpr, R_: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        """B-2: program (token chunk, D chunk): dx_s = α_s dh + ρ_s (dℓ_s q + W dP_s) − coef_s x_s for every source,
        and this chunk's partials of dW = Σ ρ x ⊗ dP (BLOCK_D, R) and dq = Σ ρ dℓ x (BLOCK_D)."""
        pid_c = tl.program_id(0)
        pid_d = tl.program_id(1)
        d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rr = tl.arange(0, R_)
        W = tl.load(w_ptr + d[:, None] * R_ + rr[None, :])                       # (BLOCK_D, R) bf16
        q = tl.load(q_ptr + d).to(tl.float32)
        dw = tl.zeros((BLOCK_D, R_), dtype=tl.float32)
        dq = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t0 in range(pid_c * T_CHUNK, pid_c * T_CHUNK + T_CHUNK, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            dh = tl.load(dh_ptr + t[:, None].to(tl.int64) * D + d[None, :]).to(tl.float32)
            for s in range(N_COMMITTED):
                off = (BT * s).to(tl.int64) * D
                dw, dq = _dx_one(committed_ptr + off, dcommitted_ptr + off, s, dh, t, d, rr, W, q, a_ptr, dl_ptr, dp_ptr,
                                 coef_ptr, rho_ptr, dw, dq, BT, D, R_)
            dw, dq = _dx_one(partial_ptr, dpartial_ptr, N_COMMITTED, dh, t, d, rr, W, q, a_ptr, dl_ptr, dp_ptr, coef_ptr,
                             rho_ptr, dw, dq, BT, D, R_)
        tl.store(dw_part_ptr + (pid_c * D + d[:, None]) * R_ + rr[None, :], dw)
        tl.store(dq_part_ptr + pid_c * D + d, dq)



# _dx_kernel launch: token chunks, tile sizes, warps, pipeline stages (best at C n=8 and 1B n=7 in the kernel sweep).
DX_CFG = {"n_chunks": 128, "BLOCK_T": 32, "BLOCK_D": 256, "num_warps": 4, "num_stages": 3}
MAX_STEPS = 3


def _rw(r: int) -> int:
    return triton.next_power_of_2(r + 1)


def fused_supported(committed: torch.Tensor, partial: torch.Tensor, U: torch.Tensor, steps: int) -> bool:
    """True when the fused kernels can run this call: CUDA bf16 sources, committed (n, *partial.shape), BT % 64 == 0,
    D % 128 == 0, a power-of-two rank >= 16, at most MAX_STEPS steps."""
    if not AVAILABLE or not partial.is_cuda:
        return False
    if partial.dtype != torch.bfloat16 or committed.dtype != torch.bfloat16:
        return False
    if committed.dim() != partial.dim() + 1 or tuple(committed.shape[1:]) != tuple(partial.shape):
        return False
    D = partial.shape[-1]
    BT = partial.numel() // D
    r = U.shape[1]
    return BT % 64 == 0 and D % 128 == 0 and r >= 16 and (r & (r - 1)) == 0 and 0 <= steps <= MAX_STEPS


@torch.library.custom_op("kotodama::energy_split_fwd", mutates_args=())
def energy_split_fwd(committed: torch.Tensor, partial: torch.Tensor, qw: torch.Tensor, nw: torch.Tensor,
                     U: torch.Tensor, sigma: torch.Tensor, eps: float, steps: int
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h, ρ (S, BT), [P | ℓ] (S, BT, RW), α (S, BT)) for one energy-routed site."""
    n = committed.shape[0]
    D = partial.shape[-1]
    BT = partial.numel() // D
    R = U.shape[1]
    RW = _rw(R)
    S = n + 1
    S_PAD = max(2, triton.next_power_of_2(S))
    dev = partial.device
    committed = committed.reshape(n, BT, D).contiguous()
    part = partial.reshape(BT, D).contiguous()
    wq = torch.zeros(D, RW, device=dev, dtype=part.dtype)
    wq[:, :R] = (nw.float().unsqueeze(-1) * U.float()).to(part.dtype)
    wq[:, R] = qw.to(part.dtype)
    rho = torch.empty(S, BT, device=dev, dtype=torch.float32)
    pl = torch.empty(S, BT, RW, device=dev, dtype=torch.float32)
    if n:
        _terms_kernel[(BT // 64, n)](committed, wq, rho, pl, 0, BT, eps, D=D, R_=R, RW_=RW, BLOCK_T=64, BLOCK_D=128)
    _terms_kernel[(BT // 64, 1)](part, wq, rho, pl, n, BT, eps, D=D, R_=R, RW_=RW, BLOCK_T=64, BLOCK_D=128)
    a = torch.empty(S, BT, device=dev, dtype=torch.float32)
    h = torch.empty(BT, D, device=dev, dtype=part.dtype)
    _steps_mix_kernel[(BT // 16,)](committed, part, pl, sigma, a, h, BT, D=D, N_COMMITTED=n, S=S, S_PAD=S_PAD, R_=R,
                                   RW_=RW, K=steps, BLOCK_T=16, BLOCK_D=128)
    return h.reshape(partial.shape), rho, pl, a


@energy_split_fwd.register_fake
def _(committed, partial, qw, nw, U, sigma, eps, steps):
    S = committed.shape[0] + 1
    BT = partial.numel() // partial.shape[-1]
    RW = 1 << (U.shape[1]).bit_length()
    f32 = dict(device=partial.device, dtype=torch.float32)
    return (partial.new_empty(partial.shape), torch.empty(S, BT, **f32), torch.empty(S, BT, RW, **f32),
            torch.empty(S, BT, **f32))


@torch.library.custom_op("kotodama::energy_split_bwd", mutates_args=())
def energy_split_bwd(dh: torch.Tensor, committed: torch.Tensor, partial: torch.Tensor, qw: torch.Tensor,
                     nw: torch.Tensor, U: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor, pl: torch.Tensor,
                     a: torch.Tensor, steps: int
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of energy_split_fwd's h for committed, partial, qw, nw, U and sigma."""
    n = committed.shape[0]
    D = partial.shape[-1]
    BT = partial.numel() // D
    R = U.shape[1]
    RW = _rw(R)
    S = n + 1
    S_PAD = max(2, triton.next_power_of_2(S))
    dev = dh.device
    committed_c = committed.reshape(n, BT, D).contiguous()
    part = partial.reshape(BT, D).contiguous()
    dh = dh.reshape(BT, D).contiguous()
    g = torch.empty(S, BT, device=dev, dtype=torch.float32)
    _g_kernel[(BT // 16,)](committed_c, part, dh, g, BT, D=D, N_COMMITTED=n, S=S, S_PAD=S_PAD, BLOCK_T=16,
                           BLOCK_D=128)
    dl = torch.empty(S, BT, device=dev, dtype=torch.float32)
    dp = torch.empty(S, BT, R, device=dev, dtype=torch.float32)
    coef = torch.empty(S, BT, device=dev, dtype=torch.float32)
    dsig_part = torch.empty(BT // 16, R, device=dev, dtype=torch.float32)
    _steps_bwd_kernel[(BT // 16,)](pl, rho, sigma, g, dl, dp, coef, dsig_part, BT, D=D, S=S, S_PAD=S_PAD, R_=R,
                                   RW_=RW, K=steps, BLOCK_T=16)
    W = (nw.float().unsqueeze(-1) * U.float())
    cfg = DX_CFG
    bt_dx, bd_dx = cfg["BLOCK_T"], cfg["BLOCK_D"]
    while D % bd_dx:
        bd_dx //= 2
    t_chunk = max(bt_dx, BT // cfg["n_chunks"])
    while BT % t_chunk:
        t_chunk *= 2
    n_chunks = BT // t_chunk
    dcommitted = torch.empty_like(committed_c)
    dpartial = torch.empty_like(part)
    dw_part = torch.empty(n_chunks, D, R, device=dev, dtype=torch.float32)
    dq_part = torch.empty(n_chunks, D, device=dev, dtype=torch.float32)
    _dx_kernel[(n_chunks, D // bd_dx)](committed_c, part, dh, a, dl, dp, coef, rho, W.to(part.dtype), qw,
                                       dcommitted, dpartial, dw_part, dq_part, BT, t_chunk, n,
                                       D=D, R_=R, BLOCK_T=bt_dx, BLOCK_D=bd_dx,
                                       num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    dW = dw_part.sum(0)
    return (dcommitted.reshape(committed.shape), dpartial.reshape(partial.shape), dq_part.sum(0).to(qw.dtype),
            (dW * U.float()).sum(-1).to(nw.dtype), (nw.float().unsqueeze(-1) * dW).to(U.dtype),
            dsig_part.sum(0).to(sigma.dtype))


@energy_split_bwd.register_fake
def _(dh, committed, partial, qw, nw, U, sigma, rho, pl, a, steps):
    return (committed.new_empty(committed.shape), partial.new_empty(partial.shape), qw.new_empty(qw.shape),
            nw.new_empty(nw.shape), U.new_empty(U.shape), sigma.new_empty(sigma.shape))


def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    committed, partial, qw, nw, U, sigma, _eps, steps = inputs
    _h, rho, pl, a = output
    ctx.save_for_backward(committed, partial, qw, nw, U, sigma, rho, pl, a)
    ctx.steps = steps


def _backward(ctx: Any, dh: torch.Tensor, _drho: Any, _dpl: Any, _da: Any) -> tuple:
    committed, partial, qw, nw, U, sigma, rho, pl, a = ctx.saved_tensors
    grads = energy_split_bwd(dh, committed, partial, qw, nw, U, sigma, rho, pl, a, ctx.steps)
    return (*grads, None, None)


energy_split_fwd.register_autograd(_backward, setup_context=_setup_context)


def route_energy_split_fused(committed: torch.Tensor, partial: torch.Tensor, qw: torch.Tensor,
                             norm_weight: torch.Tensor, eps: float, U: torch.Tensor, sigma: torch.Tensor,
                             steps: int) -> torch.Tensor:
    """kotodama.model.energy_route.route_energy_split on the fused kernels when ``fused_supported``; otherwise the
    PyTorch path (same op)."""
    if not fused_supported(committed, partial, U, steps):
        from kotodama.model.energy_route import route_energy_split

        return route_energy_split(committed, partial, qw, norm_weight, eps, U, sigma, steps)
    return energy_split_fwd(committed, partial, qw, norm_weight, U, sigma, float(eps), int(steps))[0]
