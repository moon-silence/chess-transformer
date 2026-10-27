# Chess as a Language: what a from-scratch transformer learns from grandmaster games

A 360K-parameter transformer that has never been told the rules of chess, trained on 72,885
grandmaster games written as plain text — built as a tested Python package, a reproducible
Colab sweep, and a technical report.


---

## What this is

The model, tokenizer, loss, and training utilities are the ones I wrote from scratch for UCSC
CSE 142 (a decoder-only transformer trained on TinyStories). Not one line of the model changed.
Only the data did: instead of children's stories, it reads games like

```
1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 ... 1-0
```

and predicts the next byte. Chess is a useful testbed because, unlike English, there is a
**referee**: python-chess can say with certainty whether any move the model writes is legal.
That turns "the loss went down" into questions with exact answers.

| # | Question | How it is measured |
|:--:|---|---|
| 1 | **Tokenization** — how does the BPE vocabulary size change what the model learns? | Same model, data, and compute at vocab 257 / 512 / 1024 / 2048; compared in **bits per byte** |
| 2 | **Rules** — does a model trained only on move text learn to play legal chess? | Teacher-forced **legal-move rate** on held-out games; **legal plies** before the first error when playing alone |
| 3 | **Outcome** — can it tell from the moves who is going to win? | *Planned:* result prediction from a game prefix vs. a material-count baseline |

Vocab 257 is raw bytes plus the `<|game|>` separator (no merges), so it doubles as the
character-level reference.

---

## Verified so far

| Claim | Check | Where |
|---|---|---|
| Vectorised BPE is the CSE 142 BPE | identical merges and encodings on 40 random corpora; ~35× faster (2 MB: 3.5 s vs. ~2 min) | `test_tokenizer.py` |
| CSE 142 model files unchanged | `model.py`, `nn_utils.py`, `training_utils.py`, `lr_schedule.py` byte-identical apart from the package name | — |
| Parameter count | `V·d + L(4d² + 3d·d_ff)` exactly, ≤ 500K for every vocab in the sweep | `test_model.py` |
| Causal mask | changing future tokens leaves past logits identical | `test_model.py` |
| Cross-entropy | matches `F.cross_entropy` on logits of magnitude ~30 | `test_model.py` |
| Every training game is legal chess | each game replayed on a board; SAN re-rendered canonically; 26 invalid games dropped | `test_pgn.py` |
| No train/test leakage | 9,191 duplicate games (same game in both players' archives) removed before splitting by game | `test_end_to_end.py` |
| Evaluation is correct | an oracle model that knows the true game scores 100% legal, 100% exact, at every ply | `test_legality.py` |

Sweep results (bits per byte, legal-move rate, legal plies) will be filled in from
`results/sweep.json` after the Colab run.

---

## Three engineering decisions worth pointing at

**Losses are compared in bits per byte, not per token.** A 2048-token vocabulary packs ~2.4×
more text into each token than raw bytes do, so its per-token loss is larger even when the model
is better. Every run is converted with `bpb = (nats/token) / (bytes/token) / ln 2` using the
measured bytes-per-token of the validation split.

**Prompts use token healing.** BPE learns merges that straddle move boundaries (`"5 "`, `".N"`).
Encoding a prompt like `1.e4 e5 2.` on its own can therefore produce a token sequence the model
never saw in training, which makes a correct model look illegal. The sampler drops the last
prompt token and constrains the first generated tokens to reproduce its bytes, letting the model
choose its own tokenisation across the boundary. The oracle test is what proves this works.

**Data is replayed, not trusted.** Raw PGN from different sources disagrees on spacing,
annotations, and comments. Every game is played out move by move on a real board and its
movetext regenerated from the legal moves, so the model sees one canonical format and every
training example is guaranteed legal.

---

## Run it

```bash
git clone https://github.com/moon-silence/chess-transformer.git
cd chess-transformer
pip install -e ".[dev]"
pytest                                   # 89 tests

bash scripts/download_data.sh            # 19 grandmaster archives from pgnmentor.com
python scripts/prepare_data.py           # validate, dedupe, split, train 4 tokenizers (~15 min)
python scripts/run_sweep.py              # train + evaluate each vocab (GPU recommended)
python scripts/make_figures.py
```

The full sweep is meant for a GPU: open
[`notebooks/colab_sweep.ipynb`](notebooks/colab_sweep.ipynb) in Colab, select a T4 runtime, and
run all cells. Data and checkpoints persist on Google Drive, and results are pushed back to the
repo at the end.

## Layout

```
src/chesslm/
  model.py, nn_utils.py,           # from CSE 142, unchanged
  training_utils.py, lr_schedule.py
  _reference_tokenizer.py          # CSE 142 BPE, kept as the ground truth for tests
  tokenizer.py                     # vectorised BPE with identical output
  pgn.py                           # parse, validate, canonicalise games
  data.py                          # dedupe, split by game, tokenize, bytes/token
  train.py                         # training loop with bits-per-byte evaluation
  legality.py                      # token-healing sampler + legal-move metrics
scripts/                           # download, prepare, train, evaluate, sweep, figures
configs/base.yaml                  # the one config every sweep run shares
notebooks/colab_sweep.ipynb        # full experiment on a Colab GPU
tests/                             # 89 tests
```

## Data

72,885 unique games after validation and de-duplication, from the
[pgnmentor.com player archives](https://www.pgnmentor.com/files.html) of Anand, Aronian,
Capablanca, Carlsen, Caruana, Ding, Firouzja, Fischer, Giri, Grischuk, Karpov, Kasparov,
Kramnik, Nakamura, Nepomniachtchi, Rapport, So, Tal, and Topalov. Split 96 / 2 / 2 by game
(seed 42). At vocab 512 the training split is 15.0M tokens, comparable to the TinyStories subset
the model was designed for.

## License

MIT
