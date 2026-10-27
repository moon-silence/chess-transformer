"""Numerically-stable softmax, activation functions, and cross-entropy loss."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of the same shape summing to 1 along ``dim``.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    z = x - x_max
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    return x * torch.sigmoid(x)


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """
    # Flatten to (N, V) and (N,)
    V = logits.shape[-1]
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)

    # Numerically stable log-sum-exp from scratch.
    x_max = flat_logits.max(dim=-1, keepdim=True).values
    z = flat_logits - x_max
    logsumexp = x_max.squeeze(-1) + torch.log(torch.exp(z).sum(dim=-1))

    # Gather the logit corresponding to each target.
    target_logits = flat_logits.gather(
        dim=-1, index=flat_targets.unsqueeze(-1)
    ).squeeze(-1)

    nll = logsumexp - target_logits
    return nll.mean()
