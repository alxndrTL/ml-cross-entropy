# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import contextlib
import socket

import pytest
import torch
import torch.distributed
import torch.distributed.tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.multiprocessing.spawn import spawn as mp_spawn

from cut_cross_entropy.utils import to_full_tensor


def find_free_port() -> int:
    """
    Returns a free port on the system.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


class SimpleNetwork(nn.Module):
    """Simple neural network for testing FSDP2 interactions."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


def manual_all_reduce_gradients(model: nn.Module, world_size: int) -> None:
    """Manually reduce gradients across all ranks."""
    for param in model.parameters():
        if param.grad is not None:
            # Sum gradients across all ranks
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            # Average the gradients
            param.grad.div_(world_size)


def _target_fn_test_fsdp2_to_full_tensor(
    rank: int,
    world_size: int,
    port: int,
    test_case: str,
    dtype: torch.dtype,
    error_tol: float,
    mesh_shape: tuple[int, ...],
):
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda", rank % torch.cuda.device_count())
    )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        backend = "cpu:gloo,cuda:nccl"
    else:
        backend = "gloo"

    store = torch.distributed.TCPStore(
        "localhost", port, world_size=world_size, is_master=rank == 0
    )

    torch.distributed.init_process_group(
        backend=backend, store=store, world_size=world_size, rank=rank
    )
    store = None

    # Initialize device mesh
    if len(mesh_shape) == 1:
        device_mesh = init_device_mesh(device.type, mesh_shape)
    else:
        # For 2D meshes, specify dimension names for HSDP
        device_mesh = init_device_mesh(
            device.type, mesh_shape, mesh_dim_names=("replicate", "shard")
        )

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Network dimensions
    input_dim, hidden_dim, output_dim = 64, 128, 32
    batch_size = 16

    # Create input data and target
    x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    target = torch.randn(batch_size, output_dim, device=device, dtype=dtype)

    # Broadcast data to ensure all ranks have the same input
    torch.distributed.broadcast(x, src=0)
    torch.distributed.broadcast(target, src=0)

    # Test case 1: Manual gradient reduction (reference implementation)
    model = SimpleNetwork(input_dim, hidden_dim, output_dim).to(device).to(dtype)

    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    if test_case == "mixed_grad":
        # Apply additional gradient accumulation for mixed case
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

    # Manually reduce gradients
    manual_all_reduce_gradients(model, world_size)

    # Store gradients for comparison
    manual_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            manual_grads[name] = param.grad.clone()

        param.grad = None

    # Apply FSDP2 sharding with device mesh
    fully_shard(model, mesh=device_mesh)

    # Test case 2: Gradient only outside network forward (using to_full_tensor)
    if test_case in ("external_grad_only", "mixed_grad"):
        output = F.linear(
            x, to_full_tensor(model.linear1.weight), to_full_tensor(model.linear1.bias)
        )
        output = torch.relu(output)
        output = F.linear(
            output, to_full_tensor(model.linear2.weight), to_full_tensor(model.linear2.bias)
        )
        loss = nn.MSELoss()(output, target)
        loss.backward()

    # Test case 3: Gradient only inside network forward
    if test_case in ("internal_grad_only", "mixed_grad"):
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

    # Collect FSDP gradients
    fsdp_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            fsdp_grads[name] = (
                param.grad.full_tensor()
                if isinstance(param.grad, torch.distributed.tensor.DTensor)
                else param.grad
            ).clone()

    torch.distributed.destroy_process_group()

    # Verify gradients match
    for name in manual_grads:
        assert name in fsdp_grads, f"Parameter {name} missing in FSDP results"
        assert torch.allclose(
            manual_grads[name], fsdp_grads[name], atol=error_tol, rtol=error_tol
        ), f"Gradients don't match for {name}: max diff = {(manual_grads[name] - fsdp_grads[name]).abs().max().item()}"


@pytest.mark.parametrize("dtype,error_tol", [(torch.float32, 1e-5), (torch.float16, 1e-3)])
@pytest.mark.parametrize("mesh_shape", [(2,), (4,), (2, 2)])
def test_fsdp2_to_full_tensor_external_grad_only(
    dtype: torch.dtype,
    error_tol: float,
    mesh_shape: tuple[int, ...],
):
    """Test to_full_tensor with gradients only outside network forward method."""
    nprocs = int(torch.tensor(mesh_shape).prod().item())
    port = find_free_port()
    mp_spawn(
        _target_fn_test_fsdp2_to_full_tensor,
        args=(nprocs, port, "external_grad_only", dtype, error_tol, mesh_shape),
        nprocs=nprocs,
        join=True,
    )


@pytest.mark.parametrize("dtype,error_tol", [(torch.float32, 1e-5), (torch.float16, 1e-3)])
@pytest.mark.parametrize("mesh_shape", [(2,), (4,), (2, 2)])
def test_fsdp2_to_full_tensor_internal_grad_only(
    dtype: torch.dtype,
    error_tol: float,
    mesh_shape: tuple[int, ...],
):
    """Test to_full_tensor with gradients only inside network forward method."""
    nprocs = int(torch.tensor(mesh_shape).prod().item())
    port = find_free_port()

    # For internal grad only, we compare against the FSDP implementation
    # since manual reduction doesn't apply here
    mp_spawn(
        _target_fn_test_fsdp2_to_full_tensor,
        args=(nprocs, port, "internal_grad_only", dtype, error_tol, mesh_shape),
        nprocs=nprocs,
        join=True,
    )


@pytest.mark.parametrize("dtype,error_tol", [(torch.float32, 1e-5), (torch.float16, 1e-3)])
@pytest.mark.parametrize("mesh_shape", [(2,), (4,), (2, 2)])
def test_fsdp2_to_full_tensor_mixed_grad(
    dtype: torch.dtype,
    error_tol: float,
    mesh_shape: tuple[int, ...],
):
    """Test to_full_tensor with gradients from both outside and inside network."""
    nprocs = int(torch.tensor(mesh_shape).prod().item())
    port = find_free_port()

    # Test mixed gradients
    mp_spawn(
        _target_fn_test_fsdp2_to_full_tensor,
        args=(nprocs, port, "mixed_grad", dtype, error_tol, mesh_shape),
        nprocs=nprocs,
        join=True,
    )
