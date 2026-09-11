# ANLP Assignment 1 — Transformers from Scratch, Architectural Variants, and BLT

**Course:** Advanced Natural Language Processing — Spring 2026  
**Deadline:** September 2nd, 2026, 23:59 IST

---

## Task

Build a custom Sequence-to-Sequence Transformer to map **encrypted binary sequences → plaintext English sentences** and run a controlled ablation study across 5 architectural configurations.

---

## Directory Structure

```
ANLP_A1/
├── src/
│   ├── models/
│   │   ├── attention.py      # MHA and GQA
│   │   ├── positional.py     # Sinusoidal PE and RoPE
│   │   ├── norm.py           # LayerNorm and RMSNorm
│   │   ├── transformer.py    # Encoder, Decoder, Seq2SeqTransformer, build_model()
│   │   └── blt.py            # LocalEncoder, LocalDecoder, BLTTransformer (C5)
│   ├── dataset.py            # Tokenized (C1–C4) and Byte-level (C5) data loaders
│   ├── train.py              # Main training loop — all 5 configs
│   └── utils.py              # Metrics (Bit Acc, Seq Acc, Levenshtein, BLEU, ROUGE) + plots
├── outputs/                  # Saved checkpoints, loss curves, comparison charts
├── Dataset_A1/               # brown_cipher.txt and brown_plain.txt
├── ANLP_A1_Training.ipynb    # Self-contained Google Colab training notebook
├── Report.pdf                # Final report (6 pages max)
└── README.md                 # This file
```

---

## Ablation Configurations

| Config | Change from Base | Positional Enc. | Attention | Normalization | Tokenization |
|--------|-----------------|-----------------|-----------|---------------|--------------|
| **C1** | None (Baseline) | Sinusoidal | MHA | LayerNorm | Char-level |
| **C2** | Positional Enc. | **RoPE** | MHA | LayerNorm | Char-level |
| **C3** | Attention | Sinusoidal | **GQA** (4 KV groups) | LayerNorm | Char-level |
| **C4** | Normalization | Sinusoidal | MHA | **RMSNorm** | Char-level |
| **C5** | Tokenization | Sinusoidal | MHA | LayerNorm | **BLT (byte-level)** |

---

## Setup

```bash
pip install torch sacrebleu rouge-score wandb huggingface_hub matplotlib
```

## Training

### Local
```bash
# Train a specific config (C1, C2, C3, C4, or C5)
python src/train.py --config_name C1 --data_dir Dataset_A1/Dataset_A1

# Train all configs
for cfg in C1 C2 C3 C4 C5; do
    python src/train.py --config_name $cfg --data_dir Dataset_A1/Dataset_A1
done
```

### Google Colab (recommended — T4 GPU)
1. Upload this project folder to Google Drive at `MyDrive/ANLP_A1/`
2. Open `ANLP_A1_Training.ipynb` in Colab
3. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4`
4. Run cells top to bottom

---

## Shared Hyperparameters (all configs)

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `num_heads` | 8 |
| `num_layers` | 4 |
| `d_ff` | 1024 |
| `batch_size` | 32 |
| `learning_rate` | 1e-4 |
| `warmup_steps` | 4000 |
| `epochs` | 20 |
| `dropout` | 0.1 |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Bit Accuracy | % exact character matches |
| Sequence Accuracy | % perfectly reconstructed sequences (exact match) |
| Levenshtein Distance | Average edit distance (lower = better) |
| BLEU Score | N-gram precision (C1–C4 only) |
| ROUGE-L | Longest common subsequence F1 (C1–C4 only) |

All metrics computed using **greedy decoding** for consistency.

---

## Links

- **WandB Runs:** <!-- ADD YOUR WANDB PROJECT LINK HERE after training -->
- **HuggingFace Checkpoints:** <!-- ADD YOUR HF REPO LINK HERE after uploading -->

---

## Key Design Decisions

- **Pre-Layer Norm:** Applied before each sub-layer (not after) for stable gradients
- **Weight Tying:** Output projection shares weights with target token embedding
- **Noam LR Schedule:** Warmup for 4000 steps, then 1/√step decay
- **Teacher Forcing:** Ground-truth target prefix fed to decoder during training
- **GQA Groups:** 4 KV groups out of 8 query heads (50% KV cache reduction vs MHA)
- **BLT Patch Size:** 8 bytes per patch (512-byte cipher → 64 global patches)
