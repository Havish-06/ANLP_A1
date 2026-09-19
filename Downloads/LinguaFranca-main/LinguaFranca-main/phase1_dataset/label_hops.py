"""
phase1_dataset/label_hops.py
──────────────────────
Assign success/failure labels to each generated reasoning hop.

Key insight
───────────
Because 2WikiMultihopQA gives us the gold bridging entity for each hop
(from its `reasoning_graph` field), this is a PRESENCE CHECK — not an
extraction task.  We do NOT try to extract an entity from the hop text;
instead we ask: "does the hop text mention the gold entity we already know?"

Labeling pipeline per hop
──────────────────────────
  1. Parse <hopN>…</hopN> blocks from the generated CoT.
  2. For each hop, look up the gold entity from the reasoning graph.
  3. Run EntityMatcher (3-tier + optional LLM judge).
  4. Assign label 0 (success) or 1 (failure).
  5. Record first_fail_hop = smallest hop_idx with label 1, else None.

Output
──────
Enriches each record with:
    "hops": [
        {
          "hop_idx": int,
          "text": str,
          "bridging_entity_gold": str,
          "bridging_entity_pred": str,   ← best NP match, for inspection
          "match_method": str,
          "label": 0 | 1,
        }, ...
    ]
    "first_fail_hop": int | null
    "final_answer_correct": bool

Usage (standalone):
    python phase1_dataset/label_hops.py --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from utils.matching import EntityMatcher
from utils.wikidata_aliases import build_alias_table, load_alias_table

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_HOP_TAG_RE    = re.compile(r"<hop(\d+)>(.*?)</hop\1>", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>",  re.DOTALL | re.IGNORECASE)
_NP_RE         = re.compile(r"\b[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)*\b")


def _best_np(hop_text: str) -> str:
    """
    Extract the most likely bridging entity from hop text for logging/inspection.
    (Not used for the label decision — we use presence-check matching.)
    Picks the longest capitalized noun phrase as a heuristic.
    """
    spans = _NP_RE.findall(hop_text)
    if not spans:
        return ""
    return max(spans, key=len)


def _normalize_answer(text: str) -> str:
    """Lowercase + strip for final-answer comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Build a simple LLM judge from the already-loaded model (optional)
# ─────────────────────────────────────────────────────────────────────────────

def make_llm_judge(model, tokenizer, device="cuda"):
    """
    Returns a callable (gold_entity, hop_text) -> bool that uses the
    same 3B model as a binary judge for ambiguous cases.

    The judge is called only when all 3 matching tiers fail AND the
    LLM budget (default 5 %) has not been exhausted.
    """
    import torch

    JUDGE_PROMPT = (
        "Does the following reasoning step correctly identify the entity '{entity}'?\n"
        "Reasoning step: \"{hop}\"\n"
        "Answer with exactly 'yes' or 'no'."
    )

    @torch.no_grad()
    def judge(gold_entity: str, hop_text: str) -> bool:
        prompt = JUDGE_PROMPT.format(entity=gold_entity, hop=hop_text)
        ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out    = model.generate(ids, max_new_tokens=3, do_sample=False)
        reply  = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return reply.strip().lower().startswith("yes")

    return judge


# ─────────────────────────────────────────────────────────────────────────────
# Per-example labeling
# ─────────────────────────────────────────────────────────────────────────────

def label_example(
    example: dict,
    matcher: EntityMatcher,
) -> dict:
    """
    Annotate a single generated-CoT example with hop-level labels.

    Parameters
    ----------
    example : dict
        Must contain 'generated_cot', 'reasoning_graph', 'gold_answer',
        'predicted_answer'.
    matcher : EntityMatcher
        Configured 3-tier matcher.

    Returns
    -------
    dict
        Enriched example with 'hops', 'first_fail_hop', 'final_answer_correct'.
    """
    gen_text     = example.get("generated_cot", "")
    graph        = example.get("reasoning_graph", [])
    gold_answer  = example.get("gold_answer", "")
    pred_answer  = example.get("predicted_answer", "")

    # Parse <hopN> blocks
    parsed_hops = {}
    for m in _HOP_TAG_RE.finditer(gen_text):
        idx = int(m.group(1))
        parsed_hops[idx] = m.group(2).strip()

    labeled_hops = []
    first_fail   = None

    for node in graph:
        hop_idx      = node["hop"]
        gold_entity  = node.get("gold_entity", "")
        hop_text     = parsed_hops.get(hop_idx, "")

        if not hop_text:
            # Model didn't produce this hop tag at all → treat as failure
            labeled_hops.append({
                "hop_idx":              hop_idx,
                "text":                 "",
                "bridging_entity_gold": gold_entity,
                "bridging_entity_pred": "",
                "match_method":         "missing",
                "label":                1,
            })
            if first_fail is None:
                first_fail = hop_idx
            continue

        if not gold_entity:
            # No gold entity in the reasoning graph for this hop — skip labeling
            labeled_hops.append({
                "hop_idx":              hop_idx,
                "text":                 hop_text,
                "bridging_entity_gold": "",
                "bridging_entity_pred": _best_np(hop_text),
                "match_method":         "skipped_no_gold",
                "label":                -1,   # -1 = unlabeled
            })
            continue

        result = matcher.match(gold_entity, hop_text)
        label  = 0 if result.matched else 1

        labeled_hops.append({
            "hop_idx":              hop_idx,
            "text":                 hop_text,
            "bridging_entity_gold": gold_entity,
            "bridging_entity_pred": _best_np(hop_text),
            "match_method":         result.method,
            "sbert_score":          result.score,
            "label":                label,
        })

        if label == 1 and first_fail is None:
            first_fail = hop_idx

    # Final-answer correctness (normalized string match)
    final_correct = (
        _normalize_answer(pred_answer) == _normalize_answer(gold_answer)
        if pred_answer
        else False
    )

    return {
        **example,
        "hops":               labeled_hops,
        "first_fail_hop":     first_fail,
        "final_answer_correct": final_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_labeling(cfg: dict, llm_judge=None) -> Path:
    """
    Label all examples in generated_cot.jsonl.
    Returns path to labeled.jsonl.
    """
    raw_dir   = Path(cfg["data"]["raw_dir"])
    match_cfg = cfg["matching"]

    cot_path     = raw_dir / "2wikimultihopqa" / "generated_cot.jsonl"
    labeled_path = raw_dir / "2wikimultihopqa" / "labeled.jsonl"

    if not cot_path.exists():
        raise FileNotFoundError(f"{cot_path} not found — run generate_cot.py first.")

    if labeled_path.exists():
        logger.info("labeled.jsonl already exists — skipping labeling.")
        return labeled_path

    # Load generated examples
    with open(cot_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    # Build alias table
    all_entities = list({
        node["gold_entity"]
        for ex in examples
        for node in ex.get("reasoning_graph", [])
        if node.get("gold_entity")
    })
    logger.info("Building Wikidata alias table for %d unique entities …", len(all_entities))
    aliases = build_alias_table(
        all_entities,
        match_cfg["wikidata_aliases_path"],
        rate_limit_sec=match_cfg.get("sparql_rate_limit_sec", 0.5),
    )

    matcher = EntityMatcher(
        aliases=aliases,
        sbert_model_name=match_cfg["sbert_model"],
        sbert_threshold=match_cfg["sbert_threshold"],
        llm_judge=llm_judge,
        llm_budget=match_cfg.get("llm_fallback_budget", 0.05),
    )

    # Label all examples
    labeled = [label_example(ex, matcher) for ex in tqdm(examples, desc="Labeling hops")]

    # Log statistics
    total_hops = sum(len(ex["hops"]) for ex in labeled)
    fail_hops  = sum(
        1 for ex in labeled
        for h in ex["hops"] if h["label"] == 1
    )
    logger.info(
        "Labeled %d examples  |  %d hops total  |  %d failures (%.1f%%)",
        len(labeled), total_hops, fail_hops, 100 * fail_hops / max(total_hops, 1),
    )
    logger.info("Matcher stats: %s", matcher.stats())

    # Save
    with open(labeled_path, "w", encoding="utf-8") as f:
        for ex in labeled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Saved labeled examples → %s", labeled_path)
    return labeled_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Label hop success/failure")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_labeling(config)
