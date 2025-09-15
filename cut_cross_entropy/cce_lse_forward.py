# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_softcapping


def _cce_lse_forward_kernel(
    E,
    C,
    Bias,
    SumExp,
    MaxVal,
    LA,
    CorrectLogit,
    Locks,
    Valids,
    Targets,
    softcap,
    shift,
    B,
    V,
    D,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_pid_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_pid_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    accum = accum.to(dtype=E.dtype.element_ty)
    if HAS_BIAS:
        bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
        accum += bias[None, :]
    accum = accum.to(dtype=tl.float32)

    logits = tl.where(offs_v[None, :] < V, accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    if HAS_LA:
        this_avg_logit = tl.sum(logits, 0) / B
        tl.atomic_add(LA + offs_v, this_avg_logit, mask=offs_v < V)

    if HAS_TARGETS:
        if HAS_SHIFT:
            target_offs_b = offs_b + shift
        else:
            target_offs_b = offs_b

        this_targets = tl.load(Targets + target_offs_b, mask=target_offs_b < BMax, other=V + 1)

        offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)

        correct_logit_ptrs = CorrectLogit + offs_b

        correct_logit_ptrs = tl.broadcast_to(correct_logit_ptrs[:, None], (BLOCK_B, BLOCK_V))
        tl.store(correct_logit_ptrs, logits, mask=this_targets[:, None] == offs_v[None, :])
    else:
        offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)

    this_mx = tl.max(logits, axis=1)

    o_mask = offs_b < B

    sum_exp_ptrs = SumExp + offs_b
    max_val_ptrs = MaxVal + offs_b

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    sum_exp = tl.load(sum_exp_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    old_max = tl.load(max_val_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")

    max_val = tl.maximum(old_max, this_mx)

    this_sum_exp = tl.sum(tl.exp(logits - max_val[:, None]), axis=1)

    sum_exp = sum_exp * tl.exp(old_max - max_val) + this_sum_exp

    tl.store(sum_exp_ptrs, sum_exp, mask=o_mask, eviction_policy="evict_last")
    tl.store(max_val_ptrs, max_val, mask=o_mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(this_locks, 0)


_cce_lse_forward_kernel = triton.jit(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 8,
        "DOT_PRECISION": lambda args: "tf32"
        if torch.get_float32_matmul_precision() == "high"
        else "ieee",
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
    }
)(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = cce_forward_autotune()(_cce_lse_forward_kernel)  # type: ignore


@dataclass(slots=True)
class LSEReturn:
    lse: torch.Tensor
    logit_avg: torch.Tensor | None
    correct_logit: torch.Tensor | None


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    targets: torch.Tensor | None = None,
    shift: int = 0,
    return_logit_avg: bool = False,
) -> LSEReturn:
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    if bias is not None:
        assert bias.ndim == 1
        assert c.shape[0] == bias.shape[0]

    V, D = c.shape
    # Allocates output.
    sum_exp = e.new_full((B,), 0.0, dtype=torch.float32)
    max_val = e.new_full((B,), -torch.inf, dtype=torch.float32)
    correct_logit = e.new_full((B,), 0.0, dtype=torch.float32) if targets is not None else None
    assert sum_exp.stride(0) == 1
    assert max_val.stride(0) == 1

    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.uint32,
    )

    if return_logit_avg:
        logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)
    else:
        logit_avg = None

    # 1D launch kernel where each block gets its own program.
    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        bias,
        sum_exp,
        max_val,
        logit_avg,
        correct_logit,
        locks,
        valids,
        targets,
        softcap,
        shift,
        B,
        V,
        D,  #
        e.size(0),
        e.stride(0),
        e.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    lse = sum_exp.log() + max_val

    return LSEReturn(lse, logit_avg, correct_logit)
