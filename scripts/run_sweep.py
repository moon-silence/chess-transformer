"""Train + evaluate every vocabulary size and collect results/sweep.json.

    python scripts/run_sweep.py                      # full sweep (Colab GPU)
    python scripts/run_sweep.py --vocab-sizes 257 512 --max_steps 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chesslm.train import TrainConfig, train
from evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[257, 512, 1024, 2048])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results", default="results/sweep.json")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--n-positions", type=int, default=500)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config, max_steps=args.max_steps)
    rows = []
    for v in args.vocab_sizes:
        s = train(cfg, f"{args.data_dir}/v{v}", f"{args.runs_dir}/v{v}")
        e = evaluate(v, args.data_dir, args.runs_dir, n_positions=args.n_positions)
        rows.append({**{k: s[k] for k in s if k != "history"}, "history": s["history"],
                     "legal_rate": e["teacher_forced"]["legal_rate"],
                     "exact_match_rate": e["teacher_forced"]["exact_match_rate"],
                     "legal_by_phase": e["teacher_forced"]["by_phase"],
                     "mean_free_legal_plies": e["free_running"]["mean_legal_plies"],
                     "sample": e["sample_from_1.e4_e5"]})
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results).write_text(json.dumps(rows, indent=2))
    print("\nvocab  bits/byte  legal-rate  free-legal-plies")
    for r in rows:
        print(f"{r['vocab_size']:5d}  {r['best_val_bpb']:.4f}     {r['legal_rate']:.3f}"
              f"       {r['mean_free_legal_plies']:.1f}")


if __name__ == "__main__":
    main()
