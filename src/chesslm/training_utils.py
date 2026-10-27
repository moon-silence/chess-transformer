"""Training utilities: batching and text generation."""

from __future__ import annotations

import torch

from chesslm.nn_utils import softmax


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from a 1-D token array.

    Args:
        data: 1-D tensor of token IDs.
        batch_size: Number of examples per batch.
        context_length: Number of tokens in each sequence.
        device: Device to place tensors on.

    Returns:
        ``(x, y)`` both of shape ``(batch_size, context_length)``.
    """
    n = len(data)
    if n <= context_length:
        raise ValueError(
            f"data length ({n}) must be greater than context_length "
            f"({context_length})"
        )
    # Random start indices in [0, n - context_length), with replacement.
    starts = torch.randint(0, n - context_length, (batch_size,))
    x = torch.stack([data[s : s + context_length] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + context_length] for s in starts])
    x = x.to(device=device, dtype=torch.long)
    y = y.to(device=device, dtype=torch.long)
    return x, y


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from a language model.

    Args:
        model: Maps ``(B, T)`` integer input to ``(B, T, vocab_size)`` logits.
        prompt_ids: Starting token IDs.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        context_length: Maximum context window (defaults to ``model.context_length``).

    Returns:
        List of token IDs (prompt + generated).
    """
    device = next(model.parameters()).device
    if context_length is None:
        context_length = model.context_length

    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        # Feed at most the last context_length tokens.
        context = ids[-context_length:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)  # (1, T, V)
        next_logits = logits[:, -1, :] / temperature  # (1, V)
        probs = softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(int(next_id))
    return ids
