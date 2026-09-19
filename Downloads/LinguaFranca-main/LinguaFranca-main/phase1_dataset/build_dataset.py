"""
phase1_dataset/build_dataset.py
─────────────────────────
Orchestration script — runs the full Phase 1 pipeline end-to-end:

  Step 1  download.py        — download 2WikiMultihopQA + HotpotQA
  Step 2  generate_cot.py    — run LLM → XML-tagged CoT + hidden states
  Step 3  label_hops.py      — per-hop success/failure labels (presence check)
  Step 4  counterfactuals.py — balance failure class with minimal edits
  Step 5  split & write      — 60/20/20 JSONL split by question ID

Outputs
───────
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
  data/processed/hotpotqa_test.jsonl   (OOD eval, sampled)

Usage:
  python phase1_dataset/build_dataset.py --config configs/data_config.yaml
  python phase1_dataset/build_dataset.py --config configs/data_config.yaml --dry-run
  python phase1_dataset/build_dataset.py --config configs/data_config.yaml --skip-download
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test splitter
# ─────────────────────────────────────────────────────────────────────────────

def split_by_id(
    examples: list[dict],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split examples into train/val/test by question ID.

    Why by ID and not by index?
    ───────────────────────────
    A counterfactual example has id = "<base_id>_cf".
    We must ensure the base example and its counterfactual end up in the
    SAME split to prevent data leakage (a probe seeing the clean version in
    train and evaluating on its counterfactual in test would be cheating).

    Algorithm:
      1. Collect all *base* IDs (stripping _cf suffix).
      2. Shuffle and partition base IDs into train/val/test.
      3. Assign each example to the split of its base ID.
    """
    base_ids: list[str] = sorted({
        ex["id"].removesuffix("_cf") for ex in examples
    })
    rng.shuffle(base_ids)

    n_train = int(len(base_ids) * train_frac)
    n_val   = int(len(base_ids) * val_frac)

    train_ids = set(base_ids[:n_train])
    val_ids   = set(base_ids[n_train : n_train + n_val])
    # rest → test

    train, val, test = [], [], []
    for ex in examples:
        base = ex["id"].removesuffix("_cf")
        if base in train_ids:
            train.append(ex)
        elif base in val_ids:
            val.append(ex)
        else:
            test.append(ex)

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────────────────────────────────────

def check_class_balance(split_name: str, examples: list[dict]) -> None:
    """Warn if failure ratio is outside [0.30, 0.70]."""
    total = len(examples)
    if total == 0:
        return
    n_fail  = sum(1 for ex in examples if ex.get("first_fail_hop") is not None)
    ratio   = n_fail / total
    status  = "✓" if 0.30 <= ratio <= 0.70 else "⚠ IMBALANCED"
    logger.info(
        "  %s  %s: %d examples, failure ratio=%.1f%%",
        status, split_name, total, 100 * ratio,
    )


def check_no_leakage(
    train: list[dict],
    val:   list[dict],
    test:  list[dict],
) -> None:
    """Assert no question ID appears in more than one split."""
    train_ids = {ex["id"] for ex in train}
    val_ids   = {ex["id"] for ex in val}
    test_ids  = {ex["id"] for ex in test}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    for overlap, name in [(tv, "train∩val"), (tt, "train∩test"), (vt, "val∩test")]:
        if overlap:
            raise RuntimeError(f"Data leakage detected in {name}: {list(overlap)[:5]}")
    logger.info("  ✓ No ID leakage between splits.")


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("  Saved %d records → %s", len(examples), path)


# ─────────────────────────────────────────────────────────────────────────────
# HotpotQA OOD sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_hotpotqa(cfg: dict, rng: random.Random) -> list[dict]:
    """
    Sample a small held-out set from HotpotQA for OOD evaluation.
    Only final-answer correctness is evaluated (no hop labeling).
    """
    raw_path = (
        Path(cfg["data"]["raw_dir"]) / "hotpotqa" / "validation.jsonl"
    )
    if not raw_path.exists():
        logger.warning("HotpotQA validation JSONL not found — skipping OOD set.")
        return []

    with open(raw_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    rng.shuffle(records)
    n_ood = 500   # fixed OOD eval size
    sampled = records[:n_ood]

    return [
        {
            "id":           rec.get("id", f"hotpot_{i}"),
            "source":       "hotpotqa",
            "question":     rec["question"],
            "gold_answer":  rec["answer"],
            "context":      rec.get("context", {}),
        }
        for i, rec in enumerate(sampled)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, dry_run: bool = False, skip_download: bool = False) -> None:
    seed   = cfg.get("seed", 42)
    rng    = random.Random(seed)
    splits = cfg["data"]["splits"]

    processed_dir = Path(cfg["data"]["processed_dir"])

    # ── Step 1: Download ────────────────────────────────────────────────────
    if not skip_download:
        logger.info("═══ Step 1: Downloading datasets ═══")
        from phase1_dataset.download import download_all
        download_all(cfg)
    else:
        logger.info("Skipping download (--skip-download).")

    # ── Step 2: CoT generation + hidden states ──────────────────────────────
    logger.info("═══ Step 2: Generating CoT trajectories ═══")
    from phase1_dataset.generate_cot import run_generation
    run_generation(cfg, dry_run=dry_run)

    # ── Step 3: Hop-level labeling ──────────────────────────────────────────
    logger.info("═══ Step 3: Labeling hops ═══")
    from phase1_dataset.label_hops import run_labeling
    run_labeling(cfg)

    # ── Step 4: Counterfactual construction ─────────────────────────────────
    if not dry_run:
        logger.info("═══ Step 4: Building counterfactuals ═══")
        from phase1_dataset.counterfactuals import run_counterfactuals
        augmented_path = run_counterfactuals(cfg)
    else:
        logger.info("[DRY RUN] Skipping counterfactual generation.")
        augmented_path = (
            Path(cfg["data"]["raw_dir"]) / "2wikimultihopqa" / "labeled.jsonl"
        )

    # ── Step 5: Split and write ─────────────────────────────────────────────
    logger.info("═══ Step 5: Splitting and writing processed files ═══")

    with open(augmented_path, encoding="utf-8") as f:
        all_examples = [json.loads(line) for line in f]

    train, val, test = split_by_id(
        all_examples,
        train_frac=splits["train"],
        val_frac=splits["val"],
        rng=rng,
    )

    # Quality checks
    logger.info("Class balance check:")
    check_class_balance("train", train)
    check_class_balance("val",   val)
    check_class_balance("test",  test)
    check_no_leakage(train, val, test)

    # Write 2Wiki splits
    save_jsonl(train, processed_dir / "train.jsonl")
    save_jsonl(val,   processed_dir / "val.jsonl")
    save_jsonl(test,  processed_dir / "test.jsonl")

    # Write HotpotQA OOD set
    hotpotqa_examples = sample_hotpotqa(cfg, rng)
    if hotpotqa_examples:
        save_jsonl(hotpotqa_examples, processed_dir / "hotpotqa_test.jsonl")

    # ── Final summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════")
    logger.info("  Phase 1 complete!")
    logger.info("  train:   %5d examples", len(train))
    logger.info("  val:     %5d examples", len(val))
    logger.info("  test:    %5d examples", len(test))
    if hotpotqa_examples:
        logger.info("  hotpotqa:%5d examples", len(hotpotqa_examples))
    logger.info("══════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Build the LinguaFranca Phase 1 dataset end-to-end"
    )
    parser.add_argument("--config",        default="configs/data_config.yaml")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Validate schema/logic without running the model")
    parser.add_argument("--skip-download", action="store_true",
                        help="Assume raw data already downloaded")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    build_dataset(config, dry_run=args.dry_run, skip_download=args.skip_download)
