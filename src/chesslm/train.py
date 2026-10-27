"""Training loop.

Identical in spirit to the CSE 142 staff ``train.py`` (AdamW, cosine LR with
warmup, gradient clipping, fixed seed), with one addition that matters for a
tokenizer comparison: loss is also reported in **bits per byte**.

Cross-entropy per token is not comparable across vocabularies -- a 2048-token
vocabulary packs more text into each token, so each prediction is harder but
there are fewer of them. Bits per byte normalises that away:

    bpb = (nats per token) * (tokens per byte) / ln 2

and is the number every vocabulary is ranked on.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml

from chesslm.data import load_tokens
from chesslm.lr_schedule import lr_schedule
from chesslm.model import TransformerLM
from chesslm.nn_utils import cross_entropy_loss
from chesslm.training_utils import get_batch

PARAM_BUDGET = 500_000


@dataclass
class TrainConfig:
    d_model: int = 64
    n_layers: int = 5
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    context_length: int = 256
    max_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        names = {f.name for f in fields(cls)}
        raw = {k: v for k, v in raw.items() if k in names}
        raw.update({k: v for k, v in overrides.items() if v is not None and k in names})
        return cls(**raw)


def count_params(model: torch.nn.Module) -> int:
    """Deduplicated by storage, so the tied lm_head/embedding counts once."""
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def build_model(cfg: TrainConfig, vocab_size: int) -> TransformerLM:
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    )


@torch.no_grad()
def estimate_loss(model, data, cfg: TrainConfig, device, iters: int) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, cfg.batch_size, cfg.context_length, device)
        total += cross_entropy_loss(model(x), y).item()
    model.train()
    return total / iters


def nats_to_bpb(nats_per_token: float, bytes_per_token: float) -> float:
    return nats_per_token / bytes_per_token / math.log(2)


def train(
    cfg: TrainConfig,
    data_dir: str | Path,
    out_dir: str | Path,
    device: str | None = None,
    log=print,
) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    meta = json.loads((data_dir / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    bpt_val = meta["splits"]["val"]["bytes_per_token"]
    train_data = load_tokens(data_dir / "train.bin")
    val_data = load_tokens(data_dir / "val.bin")

    model = build_model(cfg, vocab_size).to(device)
    n_params = count_params(model)
    n_embed = vocab_size * cfg.d_model
    log(
        f"vocab {vocab_size} | params {n_params:,} (embedding {n_embed:,}, "
        f"blocks {n_params - n_embed:,}) | device {device}"
    )
    if n_params > PARAM_BUDGET:
        log(f"WARNING: {n_params:,} parameters exceeds the {PARAM_BUDGET:,} budget")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    history: list[dict] = []
    best_val = float("inf")
    t0 = time.time()
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val = estimate_loss(model, val_data, cfg, device, cfg.eval_iters)
            row = {
                "step": step,
                "val_loss": val,
                "val_bpb": nats_to_bpb(val, bpt_val),
                "lr": lr_schedule(
                    step, cfg.max_steps, cfg.learning_rate, cfg.min_lr, cfg.warmup_steps
                ),
                "elapsed_s": time.time() - t0,
            }
            history.append(row)
            log(
                f"step {step:5d} | val {val:.4f} nats/tok | {row['val_bpb']:.4f} bits/byte"
                f" | {row['elapsed_s']:.0f}s"
            )
            if val < best_val:
                best_val = val
                torch.save(
                    {"model": model.state_dict(), "config": asdict(cfg), "vocab_size": vocab_size,
                     "step": step},
                    out_dir / "best.pt",
                )
        if step == cfg.max_steps:
            break

        for g in opt.param_groups:
            g["lr"] = lr_schedule(
                step, cfg.max_steps, cfg.learning_rate, cfg.min_lr, cfg.warmup_steps
            )
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, device)
        loss = cross_entropy_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        opt.step()
        if history and history[-1]["step"] == step:
            history[-1]["train_loss"] = loss.item()

    summary = {
        "vocab_size": vocab_size,
        "params": n_params,
        "embedding_params": n_embed,
        "best_val_loss": best_val,
        "best_val_bpb": nats_to_bpb(best_val, bpt_val),
        "bytes_per_token": bpt_val,
        "context_bytes": cfg.context_length * bpt_val,
        "train_seconds": time.time() - t0,
        "config": asdict(cfg),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_checkpoint(path: str | Path, device: str = "cpu") -> TransformerLM:
    ckpt = torch.load(path, map_location=device)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
