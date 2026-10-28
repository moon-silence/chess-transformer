"""Copy trained weights into models/ so the Streamlit app works from a fresh clone.

    python scripts/export_models.py              # every runs/v*/best.pt
    python scripts/export_models.py --vocab 512

For each vocabulary writes models/v{V}/{model.pt, tokenizer.json, board_probe.pt}
(the probe only if scripts/run_probes.py has been run), plus models/sample_games.jsonl:
a few hundred held-out test games the app uses as openings. Each model is ~1.5-2 MB.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def export(vocab: int, runs: Path, data: Path, results: Path, out: Path) -> Path:
    ckpt = torch.load(runs / f"v{vocab}" / "best.pt", map_location="cpu")
    dest = out / f"v{vocab}"
    dest.mkdir(parents=True, exist_ok=True)
    slim = {k: ckpt[k] for k in ("model", "config", "vocab_size", "step")}
    slim["model"] = {k: v.to(torch.float32).contiguous() for k, v in slim["model"].items()}
    torch.save(slim, dest / "model.pt")
    shutil.copy(data / f"v{vocab}" / "tokenizer.json", dest / "tokenizer.json")
    probe = results / "probes" / f"board_probe_v{vocab}.pt"
    if probe.exists():
        shutil.copy(probe, dest / "board_probe.pt")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, nargs="*", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="models")
    ap.add_argument("--n-sample-games", type=int, default=300)
    args = ap.parse_args()
    runs, data, out = Path(args.runs_dir), Path(args.data_dir), Path(args.out)
    vocabs = args.vocab or sorted(int(p.parent.name[1:]) for p in runs.glob("v*/best.pt"))
    if not vocabs:
        raise SystemExit(f"no checkpoints found under {runs}/v*/best.pt")
    for v in vocabs:
        dest = export(v, runs, data, Path(args.results_dir), out)
        size = sum(f.stat().st_size for f in dest.iterdir()) / 1e6
        print(f"vocab {v}: {', '.join(sorted(f.name for f in dest.iterdir()))} ({size:.1f} MB)")
    test = data / "games" / "test.jsonl"
    if test.exists():
        keep = ("sans", "movetext", "result", "white", "black", "white_elo", "black_elo")
        with open(test) as f, open(out / "sample_games.jsonl", "w") as g:
            for _, line in zip(range(args.n_sample_games), f, strict=False):
                row = json.loads(line)
                g.write(json.dumps({k: row.get(k) for k in keep}) + "\n")
        print(f"wrote {out / 'sample_games.jsonl'}")


if __name__ == "__main__":
    main()
