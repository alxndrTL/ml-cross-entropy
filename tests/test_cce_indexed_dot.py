# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from collections.abc import Callable

import pytest
import torch

from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def cce_lse_kernel_indexed_dot(
    e: torch.Tensor,
    c: torch.Tensor,
    inds: torch.Tensor,
    bias: torch.Tensor | None = None,
    shift: int = 0,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    lse_return = cce_lse_forward_kernel(e, c, bias, valids, softcap, inds, shift)
    assert lse_return.neg_correct_logit is not None

    return lse_return.neg_correct_logit.to(out_dtype)


@skip_no_cuda
@pytest.mark.parametrize(
    "dtype,error_tol", [(torch.float32, 5e-6), (torch.float16, 2.5e-3), (torch.bfloat16, 2.5e-2)]
)
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("shape", [(256, 512, 512), (255, 507, 512), (255, 507, 497)])
@pytest.mark.parametrize("fn", [cce_lse_kernel_indexed_dot, indexed_neg_dot_forward_kernel])
def test_indexed_dot(
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    has_bias: bool,
    shape: tuple[int, int, int],
    fn: Callable[..., torch.Tensor],
):
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    if has_bias:
        bias = torch.randn(V, device="cuda", dtype=dtype)
    else:
        bias = None

    inds = torch.randint(0, V, size=(N,), device="cuda")

    gt = e.float() @ c.float().T

    if bias is not None:
        gt += bias.float()

    if softcap is not None:
        gt = softcapping(gt, softcap)

    gt = -gt.gather(dim=1, index=inds.view(-1, 1)).view(-1)

    ref = e @ c.T

    if bias is not None:
        ref += bias

    if softcap is not None:
        ref = softcapping(ref, softcap)

    ref = -ref.gather(dim=1, index=inds.view(-1, 1)).view(-1)

    cce_neg_dot = fn(e, c, inds, bias=bias, softcap=softcap)

    expected_error = (gt - ref.float()).abs()
    cce_error = (gt - cce_neg_dot.float()).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{(cce_error - expected_error).relu().max()=}"
