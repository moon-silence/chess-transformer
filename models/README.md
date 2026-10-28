# models/

Trained weights for the Streamlit app, written by `scripts/export_models.py` at the end of
`notebooks/colab_sweep.ipynb`. One folder per tokenizer vocabulary:

    v{V}/model.pt          weights + config (no optimizer state), ~1.5-2 MB
    v{V}/tokenizer.json    the BPE tokenizer the model was trained with
    v{V}/board_probe.pt    best-layer linear board probe (from scripts/run_probes.py)
    sample_games.jsonl     300 held-out test games used as openings in the app

Empty until the Colab sweep has been run.
