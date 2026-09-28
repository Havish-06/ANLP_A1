"""
phase1_dataset/layer_probe.py
─────────────────────────────
Train a logistic probe per transformer layer on pooled hidden states
and report held-out test accuracy to identify the most informative
layers for hop-failure prediction.

Fixes the data-leakage bug in the original kaggle_phase1.ipynb Step 8,
which fit and evaluated the probe on the *same* data (inflated accuracy).
This module uses a stratified 80/20 train_test_split instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def run_layer_probe(cfg: dict, top_k: int = 8) -> list[int]:
    """
    Train a logistic probe per layer and return the top-k most informative
    layer indices (sorted ascending).

    Parameters
    ----------
    cfg     : full pipeline config dict (as loaded from data_config.yaml)
    top_k   : how many best layers to return (default 8)

    Returns
    -------
    List of layer indices (int), sorted ascending, e.g. [8, 12, 16, 20, 24, 28]
    """
    hs_dir   = Path(cfg["data"]["hidden_states_dir"])
    raw_dir  = Path(cfg["data"]["raw_dir"])
    seed     = cfg.get("seed", 42)

    labeled_path = raw_dir / "2wikimultihopqa" / "labeled.jsonl"
    if not labeled_path.exists():
        raise FileNotFoundError(f"{labeled_path} not found — run Step 7 first.")

    with open(labeled_path, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f]

    pt_files = list(hs_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt files found in {hs_dir}. "
            "Check that hidden_states.extract=true in config and Step 6 ran fully."
        )

    # Build feature matrix
    sample        = torch.load(pt_files[0], weights_only=False)
    layer_indices = sample.get("layer_indices", list(range(sample["pooled"].shape[0])))

    X_by_layer: dict = {li: [] for li in layer_indices}
    y: list = []

    for ex in examples:
        pt_path = hs_dir / (ex["id"] + ".pt")
        if not pt_path.exists():
            continue
        data   = torch.load(pt_path, weights_only=False)
        pooled = data["pooled"]
        li_list = data.get("layer_indices", list(range(pooled.shape[0])))
        for li_idx, li in enumerate(li_list):
            X_by_layer[li].append(pooled[li_idx].mean(0).detach().cpu().numpy())
        y.append(1 if ex.get("first_fail_hop") is not None else 0)

    y_arr = np.array(y)
    print(f"Features: {len(y_arr)} examples | failure ratio: {100*float(y_arr.mean()):.1f}%")

    # Train probe per layer with held-out test split
    layer_accs: dict = {}

    for li in layer_indices:
        X = np.nan_to_num(np.array(X_by_layer.get(li, [])))
        if len(X) == 0 or len(np.unique(y_arr)) < 2:
            continue

        # Stratified 80/20 split — accuracy on held-out test set only
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
        )
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_tr, y_tr)
        layer_accs[li] = accuracy_score(y_te, clf.predict(X_te))

    # Report
    print("\nLayer probe accuracy (held-out 20% test set):")
    for li, acc in sorted(layer_accs.items()):
        bar = chr(0x2588) * int(acc * 40)
        print(f"  Layer {str(li).rjust(2)} : {round(acc, 3)}  {bar}")

    best        = sorted(layer_accs, key=layer_accs.get, reverse=True)[:top_k]
    best_sorted = sorted(best)
    print(f"\n-> Suggested BEST_LAYERS = {best_sorted}")
    print("Set cfg['hidden_states']['layers'] to these and re-run Step 6.")

    return best_sorted
