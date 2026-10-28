"""Figure generation and model export, on synthetic inputs (no training needed)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from chesslm.probes import fit_linear_probe
from chesslm.tokenizer import BPETokenizer, train_bpe_from_text
from chesslm.train import TrainConfig, build_model, load_checkpoint

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _run(*args, cwd):
    return subprocess.run(
        [sys.executable, *map(str, args)], cwd=cwd, check=True, capture_output=True, text=True
    )


def _fake_probe_results(v: int) -> dict:
    layers = [0, 1, 2]
    trained = [
        {
            "layer": layer,
            "accuracy": 0.7 + 0.05 * layer,
            "occupied_accuracy": 0.3,
            "per_square_accuracy": [0.6 + 0.005 * s for s in range(64)],
        }
        for layer in layers
    ]
    fr = {"0-25%": 0.5, "25-50%": 0.55, "50-75%": 0.6, "75-100%": 0.7}
    method = {"accuracy": 0.6, "macro_f1": 0.5, "by_fraction": fr}
    return {
        "vocab_size": v,
        "layers": layers,
        "board_relative": {
            "trained": trained,
            "untrained": [{"layer": layer, "accuracy": 0.65} for layer in layers],
            "majority": {"accuracy": 0.63, "occupied_accuracy": 0.1},
        },
        "outcome": {
            k: dict(method)
            for k in ("majority", "material", "material_plus_elo", "activation_probe", "zero_shot")
        },
    }


def test_make_figures_probe_figures_only(tmp_path):
    for v in (257, 512):
        (tmp_path / f"probes_v{v}.json").write_text(json.dumps(_fake_probe_results(v)))
    out = tmp_path / "figs"
    r = _run(
        ROOT / "scripts" / "make_figures.py",
        "--results",
        tmp_path / "missing.json",
        "--probes-dir",
        tmp_path,
        "--out",
        out,
        cwd=tmp_path,
    )
    assert "skipping fig1-3" in r.stdout
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == [
        "fig4_board_probe_by_layer.png",
        "fig5_outcome_prediction.png",
        "fig6_square_accuracy.png",
    ]


def test_export_models_roundtrip(tmp_path):
    vocab, merges = train_bpe_from_text("1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 " * 20, 270, ["<|game|>"])
    tok = BPETokenizer(vocab, merges, ["<|game|>"])
    v = tok.vocab_size
    (tmp_path / "data" / f"v{v}").mkdir(parents=True)
    tok.save(tmp_path / "data" / f"v{v}" / "tokenizer.json")
    cfg = TrainConfig.from_yaml(SMOKE)
    model = build_model(cfg, v)
    (tmp_path / "runs" / f"v{v}").mkdir(parents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "vocab_size": v,
            "step": 1,
            "optimizer": {"junk": 1},
        },
        tmp_path / "runs" / f"v{v}" / "best.pt",
    )
    probe = fit_linear_probe(
        torch.randn(20, cfg.d_model).numpy(), torch.randint(0, 13, (20, 64)).numpy(), 13, epochs=2
    )
    (tmp_path / "results" / "probes").mkdir(parents=True)
    torch.save(
        {"layer": 1, **probe.state_dict()}, tmp_path / "results" / "probes" / f"board_probe_v{v}.pt"
    )

    _run(ROOT / "scripts" / "export_models.py", cwd=tmp_path)
    dest = tmp_path / "models" / f"v{v}"
    assert {p.name for p in dest.iterdir()} == {"model.pt", "tokenizer.json", "board_probe.pt"}
    assert "optimizer" not in torch.load(dest / "model.pt", map_location="cpu")
    reloaded = load_checkpoint(dest / "model.pt")
    x = torch.randint(0, v, (1, 8))
    assert torch.allclose(reloaded(x), model.eval()(x))
    assert BPETokenizer.load(dest / "tokenizer.json").encode("1.e4 e5") == tok.encode("1.e4 e5")
