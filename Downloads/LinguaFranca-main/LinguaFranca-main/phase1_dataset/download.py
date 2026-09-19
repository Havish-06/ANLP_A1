"""
phase1_dataset/download.py
─────────────────────
Download and locally cache:
  • 2WikiMultihopQA  (primary — has structured reasoning graphs)
  • HotpotQA         (out-of-domain evaluation only)

Each dataset is saved as line-delimited JSON (JSONL) in data/raw/.
Re-running is idempotent: existing files are never overwritten.

Compatibility note
──────────────────
datasets >= 3.0 removed support for custom loading scripts.
2WikiMultihopQA (xanhho/2WikiMultihopQA) uses such a script.
We try load_dataset first, fall back to direct HF Hub file downloads,
and finally try HuggingFace's automatically-generated Parquet export.

Usage (standalone):
    python phase1_dataset/download.py --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests
import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_jsonl(records, path: Path, desc: str) -> None:
    """Write an iterable of dicts to a JSONL file, skipping if it exists."""
    if path.exists():
        logger.info("  Already exists — skipping: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(records, desc=f"  Writing {desc}"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("  Saved %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Direct HuggingFace Hub Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _download_hf_hub_files(
    repo_id: str,
    out_base: Path,
) -> dict[str, Path]:
    """Fallback: Download raw JSON files directly from HuggingFace Hub repo."""
    file_mapping = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }
    out_base.mkdir(parents=True, exist_ok=True)
    paths = {}

    for split_name, filename in file_mapping.items():
        out_path = out_base / f"{split_name}.jsonl"
        if out_path.exists():
            logger.info("  Already exists — skipping: %s", out_path)
            paths[split_name] = out_path
            continue

        logger.info("Downloading %s directly from HF Hub...", filename)
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )

        with open(downloaded_file, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)

        _save_jsonl(data, out_path, f"2wiki/{split_name}")
        paths[split_name] = out_path

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Parquet fallback
# ─────────────────────────────────────────────────────────────────────────────

def _hf_parquet_urls(hf_dataset_id: str) -> dict[str, list[str]]:
    """
    Query HuggingFace's parquet export API and return
    {split_name: [url1, url2, ...]} for all available splits.

    HF auto-generates parquet exports for every public dataset.
    This lets us bypass the deprecated custom loading-script mechanism.
    """
    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={hf_dataset_id}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    split_urls: dict[str, list[str]] = {}
    for entry in data.get("parquet_files", []):
        split = entry["split"]
        split_urls.setdefault(split, []).append(entry["url"])
    return split_urls


def _download_parquet_split(
    urls: list[str],
    out_path: Path,
    split_name: str,
) -> None:
    """Download one or more parquet shards, concatenate, and save as JSONL."""
    if out_path.exists():
        logger.info("  Already exists — skipping: %s", out_path)
        return

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas pyarrow  (needed for parquet fallback)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for url in tqdm(urls, desc=f"  Downloading parquet shards for {split_name}"):
        frames.append(pd.read_parquet(url))

    df = pd.concat(frames, ignore_index=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("  Saved %d records → %s", len(df), out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2WikiMultihopQA downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_2wikimultihopqa(raw_dir: str, cfg: dict) -> dict[str, Path]:
    """
    Download 2WikiMultihopQA from Hugging Face.

    Why this dataset?
    -----------------
    It provides an explicit `evidences` field — a structured reasoning graph
    with (entity, relation, object) triples that form the gold reasoning path.
    This gives us free hop-level supervision without human annotation.

    Fields we use:
        question        — the question string
        answer          — gold final answer
        evidences       — list of (entity, relation, object) dicts — the GOLD PATH
        supporting_facts— (title, sent_id) pairs pointing to supporting passages
        context         — retrieved passages

    Compatibility
    -------------
    datasets >= 3.0 removed support for custom loading scripts.
    We try load_dataset() first; if it fails, we fall back to downloading raw JSON files
    via huggingface_hub, and finally try HF's Parquet export API.

    Returns dict mapping split name → local JSONL path.
    """
    from datasets import load_dataset

    hf_path  = cfg["data"]["datasets"]["2wikimultihopqa"]["hf_path"]
    out_base = Path(raw_dir) / "2wikimultihopqa"

    # ── Fast path: all splits already on disk ───────────────────────────────
    expected = ["train", "validation", "test"]
    if all((out_base / f"{s}.jsonl").exists() for s in expected):
        logger.info("2WikiMultihopQA already cached — skipping download.")
        return {s: out_base / f"{s}.jsonl" for s in expected}

    # ── Attempt 1: standard load_dataset ────────────────────────────────────
    logger.info("Downloading 2WikiMultihopQA from %s …", hf_path)
    try:
        ds = load_dataset(hf_path, trust_remote_code=True)   
        paths: dict[str, Path] = {} 
        for split_name, split_data in ds.items():
            out_path = out_base / f"{split_name}.jsonl"
            _save_jsonl(split_data, out_path, f"2wiki/{split_name}")
            paths[split_name] = out_path
        return paths

    except Exception as e:
        logger.warning(
            "load_dataset() failed (%s).\n"
            "Falling back to direct HuggingFace Hub JSON download …",
            e,
        )

    # ── Attempt 2: Direct HF Hub file download ──────────────────────────────
    try:
        return _download_hf_hub_files(hf_path, out_base)
    except Exception as e:
        logger.warning("Direct HF Hub download failed: %s. Trying Parquet export...", e)

    # ── Attempt 3: HuggingFace Parquet export fallback ───────────────────────
    split_urls = _hf_parquet_urls(hf_path)
    if not split_urls:
        raise RuntimeError(
            f"Failed to download dataset {hf_path}. "
            "Try pinning dependencies: pip install 'datasets<3.0.0'"
        )

    paths = {}
    for split_name, urls in split_urls.items():
        out_path = out_base / f"{split_name}.jsonl"
        _download_parquet_split(urls, out_path, split_name)
        paths[split_name] = out_path

    return paths


def download_hotpotqa(raw_dir: str, cfg: dict) -> dict[str, Path]:
    """
    Download HotpotQA (distractor setting) from Hugging Face.

    Why distractor?
    ---------------
    The distractor setting provides 10 paragraphs (2 gold + 8 distractors),
    which is a harder retrieval scenario and closer to real-world conditions.

    Fields we use:
        question         — the question string
        answer           — gold final answer
        supporting_facts — (title, sent_id) pairs for gold passages
        context          — (title, sentences) for all 10 passages

    Note: HotpotQA has NO structured reasoning graph, so we cannot do
    hop-level labeling here.  It is used only for final-answer accuracy.

    Returns dict mapping HF split name → local JSONL path.
    """
    from datasets import load_dataset

    hf_path   = cfg["data"]["datasets"]["hotpotqa"]["hf_path"]
    hf_config = cfg["data"]["datasets"]["hotpotqa"]["hf_config"]
    out_base  = Path(raw_dir) / "hotpotqa"

    logger.info("Downloading HotpotQA (%s) from %s …", hf_config, hf_path)
    ds = load_dataset(hf_path, hf_config, trust_remote_code=True)

    paths: dict[str, Path] = {}
    for split_name, split_data in ds.items():
        out_path = out_base / f"{split_name}.jsonl"
        _save_jsonl(split_data, out_path, f"hotpotqa/{split_name}")
        paths[split_name] = out_path

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def download_all(cfg: dict) -> dict[str, dict[str, Path]]:
    raw_dir = cfg["data"]["raw_dir"]
    return {
        "2wiki":    download_2wikimultihopqa(raw_dir, cfg),
        "hotpotqa": download_hotpotqa(raw_dir, cfg),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = download_all(config)
    for ds_name, splits in result.items():
        for split, path in splits.items():
            print(f"  {ds_name}/{split:10s}  →  {path}")