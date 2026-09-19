"""
phase1_dataset/counterfactuals.py
────────────────────────────
Construct minimal counterfactual questions to balance the failure class.

Why we need this
────────────────
A competent 3B model gets ~87 % of hops correct, so the natural failure rate
is low.  A probe trained on imbalanced data trivially predicts "success" for
everything.  We manufacture failures by substituting one surface entity in the
question with another that leads the model to the *wrong* bridging entity.

Why token-length matching is mandatory
──────────────────────────────────────
In Phase 2 (activation patching) we need the clean and corrupted prompts to
be token-for-token identical *up to* the substituted entity, so the hidden-
state patch lands at the exact same position.  If lengths differ, all
subsequent token positions shift and the patch hits the wrong place.

Algorithm
─────────
For each clean example where *all* hops succeed:
  1. Identify the "entry entity" — the surface form in the question that
     triggers hop 1 (e.g., "Inception" triggers retrieval of the director).
  2. Find candidate substitutes from the dataset vocabulary with the same
     token count as the original entity.
  3. Replace, re-run the model (or predict failure from the reasoning graph),
     validate that hop 1 is now labeled FAILURE.
  4. Discard any substitution where the model accidentally still gets it right.

Dual purpose
────────────
The (clean, corrupted) pairs produced here are stored as minimal pairs and
reused in Phase 2 (causal patching) without extra data construction.

Output
──────
Appends counterfactual records to the labeled pool.  Each record includes:
  "is_counterfactual": true
  "counterfactual_swap": {
      "original_entity": ...,
      "substituted_entity": ...,
      "token_count": ...,
      "target_fail_hop": ...
  }
  "clean_pair_id": <id of the original clean example>
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from phase1_dataset.label_hops import label_example
from utils.matching import EntityMatcher
from utils.wikidata_aliases import load_alias_table

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Entity vocabulary builder
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_vocab(examples: list[dict]) -> dict[int, list[str]]:
    """
    Collect all gold bridging entities from the dataset, grouped by
    their (rough) token count — used for token-length-matched substitution.

    We use whitespace splitting as a fast proxy for tokenization.
    The actual token-count check (with the real tokenizer) is done at
    substitution time.
    """
    vocab: dict[int, list[str]] = {}
    for ex in examples:
        for node in ex.get("reasoning_graph", []):
            ent = node.get("gold_entity", "").strip()
            if ent:
                n = len(ent.split())   # rough token count
                vocab.setdefault(n, []).append(ent)
    # De-duplicate within each bucket
    return {k: list(set(v)) for k, v in vocab.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Token-length checker
# ─────────────────────────────────────────────────────────────────────────────

def same_token_length(
    orig: str,
    subst: str,
    tokenizer,
) -> bool:
    """
    Check that `orig` and `subst` tokenize to the exact same number of tokens.

    This guarantees prefix-token alignment in the (clean, corrupted) pair:
    every token position before and after the substituted entity is identical,
    so an activation patch at position P in the corrupted prompt goes to the
    same semantic location as in the clean prompt.
    """
    orig_ids  = tokenizer.encode(orig,  add_special_tokens=False)
    subst_ids = tokenizer.encode(subst, add_special_tokens=False)
    return len(orig_ids) == len(subst_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-entity extractor
# ─────────────────────────────────────────────────────────────────────────────

def find_entry_entity(question: str, reasoning_graph: list[dict]) -> Optional[str]:
    """
    Identify the surface-form entity in the question that "triggers" hop 1.

    Strategy: the hop-1 gold entity is the *object* of the first reasoning
    step.  The *subject* entity (which appears in the question and causes the
    model to retrieve the hop-1 object) is what we want to swap.

    For 2WikiMultihopQA, the reasoning graph encodes:
        {"hop": 1, "gold_entity": "Christopher Nolan"}
    which means the question mentions some entity (e.g., "Inception") that the
    model uses to retrieve "Christopher Nolan".  We extract that trigger by
    looking for the longest noun phrase in the question that does NOT match the
    hop-1 gold entity itself.

    Fallback: if heuristics fail, return None (example is skipped).
    """
    import re
    if not reasoning_graph:
        return None

    hop1_gold = reasoning_graph[0].get("gold_entity", "").lower()

    # Candidate NPs from the question (Title-Case heuristic)
    pattern   = r"\b[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)*\b"
    candidates = re.findall(pattern, question)

    # Filter out the hop-1 gold entity itself and its substrings
    filtered = [
        c for c in candidates
        if c.lower() not in hop1_gold and hop1_gold not in c.lower()
    ]

    if not filtered:
        return None

    # Return the longest remaining NP as the most likely entry entity
    return max(filtered, key=len)


# ─────────────────────────────────────────────────────────────────────────────
# Per-example counterfactual generator
# ─────────────────────────────────────────────────────────────────────────────

def make_counterfactual(
    clean_example: dict,
    entity_vocab: dict[int, list[str]],
    tokenizer,
    matcher: EntityMatcher,
    model,
    cfg: dict,
    rng: random.Random,
) -> Optional[dict]:
    """
    Attempt to build a counterfactual for a single clean example.

    Returns a new example dict (with is_counterfactual=True) or None if no
    valid substitution could be found within `max_substitution_attempts`.
    """
    cf_cfg        = cfg["counterfactuals"]
    max_attempts  = cf_cfg.get("max_substitution_attempts", 25)
    validate_fail = cf_cfg.get("validate_induced_failure", True)

    question    = clean_example["question"]
    graph       = clean_example.get("reasoning_graph", [])
    entry_ent   = find_entry_entity(question, graph)

    if not entry_ent:
        return None

    # Find token-length-matched substitutes
    token_count = len(tokenizer.encode(entry_ent, add_special_tokens=False))
    same_len    = entity_vocab.get(token_count, [])
    candidates  = [
        e for e in same_len
        if e != entry_ent                         # not the same entity
        and e.lower() not in question.lower()     # not already in question
    ]
    rng.shuffle(candidates)

    for substitute in candidates[:max_attempts]:
        # Verify token-length match with the actual tokenizer (not the proxy)
        if not same_token_length(entry_ent, substitute, tokenizer):
            continue

        # Build counterfactual question
        cf_question = question.replace(entry_ent, substitute, 1)
        if cf_question == question:
            continue

        # If validation is enabled, run the model and check that hop 1 fails.
        # If validation is disabled (dry run), assume the substitution works.
        if validate_fail:
            induced_failure = _validate_induces_failure(
                cf_question, clean_example, matcher, model, tokenizer, cfg
            )
            if not induced_failure:
                continue

        # Build the counterfactual record
        cf = copy.deepcopy(clean_example)
        cf["id"]               = clean_example["id"] + "_cf"
        cf["question"]         = cf_question
        cf["is_counterfactual"] = True
        cf["clean_pair_id"]    = clean_example["id"]
        cf["counterfactual_swap"] = {
            "original_entity":    entry_ent,
            "substituted_entity": substitute,
            "token_count":        token_count,
            "target_fail_hop":    1,
        }
        # The gold answer and reasoning graph stay the same
        # (model is expected to fail to reach the correct bridging entity)
        return cf

    return None


def _validate_induces_failure(
    cf_question: str,
    clean_example: dict,
    matcher: EntityMatcher,
    model,
    tokenizer,
    cfg: dict,
) -> bool:
    """
    Run the model on the counterfactual question and check that hop 1 fails.
    Returns True if hop 1 is labeled failure (substitution is valid).
    """
    import torch
    from phase1_dataset.generate_cot import build_prompt, parse_hop_spans, parse_predicted_answer

    sys_prompt = cfg["generation"]["system_prompt"]
    prompt     = build_prompt(cf_question, sys_prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_text = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    hop_spans = parse_hop_spans(gen_text)
    if not hop_spans:
        return True   # No hop generated → counts as failure

    # Check hop 1
    hop1_text   = hop_spans[0]["text"] if hop_spans else ""
    gold_entity = (
        clean_example.get("reasoning_graph", [{}])[0].get("gold_entity", "")
    )
    result = matcher.match(gold_entity, hop1_text)
    return not result.matched   # True if model did NOT find the gold entity


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_counterfactuals(cfg: dict, model=None, tokenizer=None) -> Path:
    """
    Generate counterfactual examples to balance failure ratio.
    Returns path to augmented.jsonl.
    """
    raw_dir   = Path(cfg["data"]["raw_dir"])
    match_cfg = cfg["matching"]
    cf_cfg    = cfg["counterfactuals"]
    seed      = cfg.get("seed", 42)
    rng       = random.Random(seed)

    labeled_path   = raw_dir / "2wikimultihopqa" / "labeled.jsonl"
    augmented_path = raw_dir / "2wikimultihopqa" / "augmented.jsonl"

    if augmented_path.exists():
        logger.info("augmented.jsonl already exists — skipping.")
        return augmented_path

    if not labeled_path.exists():
        raise FileNotFoundError(f"{labeled_path} not found — run label_hops.py first.")

    with open(labeled_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    # Split into fully-clean examples (probe positive class)
    clean_examples = [
        ex for ex in examples
        if ex.get("first_fail_hop") is None
        and all(h["label"] == 0 for h in ex.get("hops", []))
    ]
    fail_examples = [ex for ex in examples if ex.get("first_fail_hop") is not None]

    current_fail_ratio = len(fail_examples) / max(len(examples), 1)
    target_ratio       = cf_cfg.get("target_failure_ratio", 0.45)

    logger.info(
        "Current failure ratio: %.1f%% (%d/%d) — target: %.0f%%",
        100 * current_fail_ratio, len(fail_examples), len(examples),
        100 * target_ratio,
    )

    n_needed = max(
        0,
        int(target_ratio * len(examples) / (1 - target_ratio)) - len(fail_examples),
    )
    logger.info("Need %d new counterfactual failures.", n_needed)

    # Load matching tools
    aliases = load_alias_table(match_cfg["wikidata_aliases_path"])
    matcher = EntityMatcher(
        aliases=aliases,
        sbert_model_name=match_cfg["sbert_model"],
        sbert_threshold=match_cfg["sbert_threshold"],
    )

    entity_vocab = build_entity_vocab(examples)

    # Load model if not provided (needed for validation)
    if model is None and cf_cfg.get("validate_induced_failure", True):
        from phase1_dataset.generate_cot import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(cfg)

    counterfactuals = []
    rng.shuffle(clean_examples)

    for ex in tqdm(clean_examples, desc="Building counterfactuals"):
        if len(counterfactuals) >= n_needed:
            break
        cf = make_counterfactual(
            ex, entity_vocab, tokenizer, matcher, model, cfg, rng
        )
        if cf is not None:
            counterfactuals.append(cf)

    logger.info("Generated %d counterfactual examples.", len(counterfactuals))

    # Merge and save
    all_examples = examples + counterfactuals
    rng.shuffle(all_examples)

    with open(augmented_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total     = len(all_examples)
    total_fail = sum(1 for ex in all_examples if ex.get("first_fail_hop") is not None)
    logger.info(
        "Saved %d total examples  |  failure ratio: %.1f%%  →  %s",
        total, 100 * total_fail / max(total, 1), augmented_path,
    )

    return augmented_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Generate counterfactual examples")
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_counterfactuals(config)
