"""
utils.py — Evaluation Metrics
================================

WHY THESE METRICS?
------------------
Different metrics capture different failure modes:

  Bit-Level Accuracy  → How many individual characters match exactly?
                        Good for seeing overall "closeness".

  Sequence Accuracy   → Is the WHOLE sequence correct? Very strict.
                        A model that gets 99% of chars right but misses 1
                        gets 0% sequence accuracy.

  Levenshtein Distance → How many edits (insert/delete/substitute) to go
                         from prediction to target? Lower = better.
                         Useful for measuring how "far off" partial matches are.

  BLEU Score          → Measures n-gram precision (how many predicted
                         n-grams appear in the reference). Standard in MT.

  ROUGE-L Score       → Measures longest common subsequence recall.
                         Captures whether key substrings are preserved.
"""

from typing import List
import torch


def bit_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Bit-Level (character-level) Accuracy.

    For each pair, count matching characters at the same position,
    then divide by the total target characters.

    Note: if lengths differ, extra positions count as wrong.

    Args:
        predictions: List of predicted strings.
        targets:     List of target strings.

    Returns:
        Float in [0, 1].
    """
    correct = 0
    total   = 0
    for pred, tgt in zip(predictions, targets):
        # Convert each string to its UTF-8 bytes, then to a flat list of bits
        pred_bits = ''.join(f'{byte:08b}' for byte in pred.encode('utf-8', errors='replace'))
        tgt_bits  = ''.join(f'{byte:08b}' for byte in tgt.encode('utf-8', errors='replace'))
        total   += len(tgt_bits)
        # Compare bit by bit; missing positions in pred count as wrong
        for i in range(len(tgt_bits)):
            if i < len(pred_bits) and pred_bits[i] == tgt_bits[i]:
                correct += 1
    return correct / total if total > 0 else 0.0


def sequence_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Sequence (exact match) Accuracy.

    A prediction is correct only if it is IDENTICAL to the target.

    Args:
        predictions: List of predicted strings.
        targets:     List of target strings.

    Returns:
        Float in [0, 1].
    """
    correct = sum(pred.strip() == tgt.strip() for pred, tgt in zip(predictions, targets))
    return correct / len(targets) if targets else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Uses dynamic programming. The edit distance is the minimum number of
    single-character edits (insert, delete, substitute) to transform s1 → s2.

    Time:  O(|s1| × |s2|)
    Space: O(|s1| × |s2|) — could be O(min) with optimized version

    Args:
        s1, s2: Input strings.

    Returns:
        Non-negative integer edit distance.
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = edit distance between s1[:i] and s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string to/from s2 / s1
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]          # characters match: no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete from s1
                    dp[i][j - 1],      # insert into s1
                    dp[i - 1][j - 1],  # substitute
                )
    return dp[m][n]


def avg_levenshtein(predictions: List[str], targets: List[str]) -> float:
    """Average Levenshtein distance across all pairs."""
    distances = [levenshtein_distance(p, t) for p, t in zip(predictions, targets)]
    return sum(distances) / len(distances) if distances else 0.0


def bleu_score(predictions: List[str], targets: List[str]) -> float:
    """
    Corpus-level BLEU score using sacrebleu.

    BLEU measures n-gram precision (n=1..4) between predictions and references,
    with a brevity penalty for very short predictions.

    Requires: pip install sacrebleu

    Args:
        predictions: List of predicted strings (one per sample).
        targets:     List of reference strings (one per sample).

    Returns:
        BLEU score in [0, 100].
    """
    try:
        import sacrebleu
        result = sacrebleu.corpus_bleu(predictions, [targets], tokenize='char')
        return result.score
    except ImportError:
        print("sacrebleu not installed. Run: pip install sacrebleu")
        return 0.0


def rouge_l_score(predictions: List[str], targets: List[str]) -> float:
    """
    Average ROUGE-L F1 score.

    ROUGE-L is based on the Longest Common Subsequence (LCS) between prediction
    and reference. It rewards predictions that preserve key subsequences of the
    target, regardless of word order.

    Requires: pip install rouge-score

    Args:
        predictions: List of predicted strings.
        targets:     List of target strings.

    Returns:
        ROUGE-L F1 in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = [
            scorer.score(tgt, pred)['rougeL'].fmeasure
            for pred, tgt in zip(predictions, targets)
        ]
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        print("rouge-score not installed. Run: pip install rouge-score")
        return 0.0


def compute_all_metrics(predictions: List[str], targets: List[str], tokenized: bool = True) -> dict:
    """
    Compute all metrics and return as a dict.

    Args:
        predictions: List of predicted strings.
        targets:     List of reference strings.
        tokenized:   If False, skip BLEU and ROUGE (used for BLT token-free model).

    Returns:
        dict with keys: bit_acc, seq_acc, avg_levenshtein, bleu, rouge_l
    """
    metrics = {
        'bit_acc':         bit_accuracy(predictions, targets),
        'seq_acc':         sequence_accuracy(predictions, targets),
        'avg_levenshtein': avg_levenshtein(predictions, targets),
    }
    if tokenized:
        metrics['bleu']    = bleu_score(predictions, targets)
        metrics['rouge_l'] = rouge_l_score(predictions, targets)
    return metrics


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, config_name: str, save_dir: str = 'outputs') -> None:
    """
    Plot train and validation loss curves for a single config and save to disk.

    Args:
        history:     Dict with keys 'train_loss' and 'val_loss' (lists per epoch).
        config_name: e.g. 'C1', 'C2', etc.
        save_dir:    Directory to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(save_dir, exist_ok=True)

        epochs = list(range(1, len(history['train_loss']) + 1))
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        plt.plot(epochs, history['val_loss'],   label='Val Loss',   marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title(f'{config_name} — Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f'{config_name}_loss_curve.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")


def plot_ablation_comparison(all_metrics: dict, save_dir: str = 'outputs') -> None:
    """
    Bar chart comparing all 5 configs across each metric.

    Args:
        all_metrics: Dict mapping config_name → metrics dict.
                     e.g. {'C1': {'bit_acc': 0.8, ...}, 'C2': {...}, ...}
        save_dir:    Directory to save plots.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        os.makedirs(save_dir, exist_ok=True)

        configs = list(all_metrics.keys())
        # Collect all metric keys that appear across any config
        metric_keys = []
        for m in all_metrics.values():
            for k in m:
                if k not in metric_keys:
                    metric_keys.append(k)

        n_metrics = len(metric_keys)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

        for ax, metric in zip(axes, metric_keys):
            values = [all_metrics[c].get(metric, 0) for c in configs]
            bars = ax.bar(configs, values, color=colors[:len(configs)], edgecolor='white', width=0.6)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.3)
            # Annotate bars with values
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * ax.get_ylim()[1],
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Ablation Study — All Configurations', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, 'ablation_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
