"""The CSE 142 model, re-checked in its new home."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from chesslm.lr_schedule import lr_schedule
from chesslm.nn_utils import cross_entropy_loss, softmax
from chesslm.train import PARAM_BUDGET, TrainConfig, build_model, count_params, nats_to_bpb


@pytest.mark.parametrize("vocab", [257, 512, 1024, 2048])
def test_parameter_formula_and_budget(vocab):
    cfg = TrainConfig()
    model = build_model(cfg, vocab)
    d, L, f = cfg.d_model, cfg.n_layers, cfg.d_ff
    assert count_params(model) == vocab * d + L * (4 * d * d + 3 * d * f)
    assert count_params(model) <= PARAM_BUDGET


def test_weight_tying_is_one_parameter():
    model = build_model(TrainConfig(), 512)
    assert model.lm_head.weight is model.token_emb.weight


def test_causality_future_tokens_cannot_change_past_logits():
    torch.manual_seed(0)
    cfg = TrainConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, context_length=32)
    model = build_model(cfg, 300)
    for block in model.blocks:  # undo zero-init so attention actually mixes positions
        torch.nn.init.normal_(block.attn.o_proj.weight, std=0.2)
        torch.nn.init.normal_(block.ffn.w_down.weight, std=0.2)
    x = torch.randint(0, 300, (2, 20))
    y = x.clone()
    y[:, 12:] = torch.randint(0, 300, (2, 8))
    a, b = model(x), model(y)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_overlength_input_raises():
    model = build_model(TrainConfig(context_length=16, d_model=32, n_heads=2), 300)
    with pytest.raises(AssertionError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_cross_entropy_matches_pytorch():
    torch.manual_seed(1)
    logits = torch.randn(3, 7, 50) * 30  # large values exercise numerical stability
    targets = torch.randint(0, 50, (3, 7))
    ours = cross_entropy_loss(logits, targets)
    theirs = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_softmax_is_stable():
    p = softmax(torch.tensor([1000.0, 1000.0, -1000.0]))
    assert torch.allclose(p, torch.tensor([0.5, 0.5, 0.0]))


def test_untrained_loss_is_log_vocab():
    torch.manual_seed(0)
    model = build_model(TrainConfig(), 512)
    x = torch.randint(0, 512, (4, 64))
    loss = cross_entropy_loss(model(x), torch.randint(0, 512, (4, 64)))
    assert abs(loss.item() - math.log(512)) < 0.05


def test_lr_schedule_endpoints():
    assert lr_schedule(0, 100, 1e-3, 1e-5, 10) == 0.0
    assert lr_schedule(10, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-3)
    assert lr_schedule(100, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-5)


def test_bits_per_byte():
    # one byte per token, uniform over 256 bytes -> exactly 8 bits per byte
    assert nats_to_bpb(math.log(256), 1.0) == pytest.approx(8.0)
    # same loss spread over 2-byte tokens is half the bits per byte
    assert nats_to_bpb(math.log(256), 2.0) == pytest.approx(4.0)
