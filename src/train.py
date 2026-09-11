"""
train.py — Training Loop for All 5 Configurations (C1–C5)
===========================================================
Track loss history per epoch for plotting after training.

Run a specific config by passing --config_name:
    python src/train.py --config_name C1
    python src/train.py --config_name C2
    python src/train.py --config_name C3
    python src/train.py --config_name C4
    python src/train.py --config_name C5

All configs share the same hyperparameters (d_model, num_layers, lr, etc.)
to ensure the ablation study only measures the effect of the changed component.

TRAINING CONCEPTS
-----------------
1. Teacher Forcing: decoder input = ground truth prefix (not model's own output)
2. Cross-Entropy Loss: -log(prob of correct token), ignoring <pad> positions
3. AdamW: Adam with decoupled weight decay (prevents L2 from interfering with
   the adaptive learning rate)
4. Noam Schedule: warmup then 1/sqrt(step) decay — from "Attention is All You Need"
5. Gradient Clipping: clamp gradient norm ≤ 1.0 to prevent exploding gradients
6. WandB: real-time logging of loss, LR, and test metrics per run
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_data
from utils   import compute_all_metrics
from models.transformer import build_model
from models.blt         import BLTTransformer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Run: pip install wandb")


# ---------------------------------------------------------------------------
# Shared hyperparameters (identical across C1–C5)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'd_model':      256,
    'num_heads':    8,
    'num_layers':   4,
    'd_ff':         1024,
    'max_len':      512,
    'dropout':      0.1,
    'num_kv_groups': 4,      # for C3 GQA

    'batch_size':    64,
    'lr':            3e-4,
    'warmup_steps':  1000,      # sweet spot: peak LR at epoch ~16, then 34 epochs of decay
    'epochs':        50,
    'grad_clip':     1.0,
    'label_smoothing': 0.1,
    'patience':      5,
    'max_src_len':   512,
    'max_tgt_len':   256,

    'data_dir':     'Dataset_A1/Dataset_A1',
    'config_name':  'C1',

    # BLT-specific (C5)
    'patch_size':    8,
    'd_local':       128,
    'local_heads':   4,
}


# ---------------------------------------------------------------------------
# Noam LR Schedule
# ---------------------------------------------------------------------------

class NoamScheduler:
    """
    lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
    Warmup linearly for `warmup_steps`, then decay as 1/sqrt(step).
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        self.opt     = optimizer
        self.d_model = d_model
        self.warmup  = warmup_steps
        self.step_n  = 0

    def step(self):
        self.step_n += 1
        lr = (self.d_model ** -0.5) * min(self.step_n ** -0.5,
                                          self.step_n * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def seq2seq_loss(logits, targets, pad_idx, label_smoothing=0.1):
    """Cross-entropy over all positions, ignoring pad_idx, with label smoothing."""
    B, T, V = logits.shape
    # .reshape() handles non-contiguous tensors (e.g. after slicing tgt[:, 1:])
    # .view() would fail here because slicing creates a non-contiguous tensor
    return nn.functional.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T),
        ignore_index=pad_idx, label_smoothing=label_smoothing
    )


def byte_loss(logits, targets, label_smoothing=0.1):
    """
    Cross-entropy for BLT (C5).
    logits:  (B, T_bytes_padded, 258)
    targets: (B, T_bytes_padded)
    """
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T),
        ignore_index=256, label_smoothing=label_smoothing
    )


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device, cfg, is_blt=False, scaler=None):
    model.train()
    total_loss, n = 0.0, 0
    use_amp = scaler is not None

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if is_blt:
                B = tgt.size(0)
                patch_size = model.patch_size
                bos_chunk = torch.full((B, patch_size), 256, dtype=torch.long, device=device)
                bos_chunk[:, 0] = 257
                tgt_input_c5 = torch.cat([bos_chunk, tgt[:, :-patch_size]], dim=1)
                
                logits = model(src, tgt_input_c5)
                loss   = byte_loss(logits, tgt)
            else:
                tgt_in  = tgt[:, :-1]                        # drop <eos>
                tgt_out = tgt[:, 1:]                         # drop <bos>
                logits  = model(src, tgt_in)                 # (B, T-1, vocab)
                pad_idx = 0  # <pad> is always index 0 in our Vocabulary
                loss    = seq2seq_loss(logits, tgt_out, pad_idx)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
        lr = scheduler.step()

        total_loss += loss.item()
        n += 1
        if n % 100 == 0 and WANDB_AVAILABLE:
            wandb.log({'train/step_loss': loss.item(), 'train/lr': lr})

    return total_loss / n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, is_blt=False):
    model.eval()
    total_loss, n = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        if is_blt:
            B = tgt.size(0)
            patch_size = model.patch_size
            bos_chunk = torch.full((B, patch_size), 256, dtype=torch.long, device=device)
            bos_chunk[:, 0] = 257
            tgt_input_c5 = torch.cat([bos_chunk, tgt[:, :-patch_size]], dim=1)
            
            logits = model(src, tgt_input_c5)
            loss   = byte_loss(logits, tgt)
        else:
            logits = model(src, tgt[:, :-1])
            loss   = seq2seq_loss(logits, tgt[:, 1:], 0)
        total_loss += loss.item()
        n += 1
    return total_loss / n


# ---------------------------------------------------------------------------
# Test metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_metrics(model, loader, src_vocab, tgt_vocab, device, is_blt=False, max_samples=200):
    model.eval()
    preds, targets = [], []

    for i, (src, tgt) in enumerate(loader):
        if i >= max_samples:
            break
        src = src.to(device)

        if is_blt:
            pred_bytes = model.greedy_decode(src)[0].tolist()
            pred_str   = bytes([b for b in pred_bytes if 0 <= b <= 255]).decode('utf-8', errors='replace')
            # tgt[0] is (n_patches, patch_size), flatten it
            tgt_flat = tgt[0].view(-1).tolist()
            tgt_str    = bytes([b for b in tgt_flat if 0 <= b <= 255]).decode('utf-8', errors='replace')
        else:
            pred_ids = model.greedy_decode(src, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
            pred_str = tgt_vocab.decode(pred_ids.tolist())
            tgt_str  = tgt_vocab.decode(tgt[0].tolist())

        preds.append(pred_str)
        targets.append(tgt_str)

    # We can pass tokenized=True even for BLT, because we want BLEU and ROUGE!
    metrics = compute_all_metrics(preds, targets, tokenized=True)
    
    if is_blt:
        # User requested to remove BLEU and ROUGE-L for C5 (BLT)
        metrics.pop('bleu', None)
        metrics.pop('rouge_l', None)

    # Log sample predictions as a WandB Table
    if WANDB_AVAILABLE:
        cols   = ['sample', 'prediction', 'target', 'match']
        rows   = [[i+1, p, t, '✅' if p.strip()==t.strip() else '❌']
                  for i, (p, t) in enumerate(zip(preds[:10], targets[:10]))]
        metrics['_pred_table'] = wandb.Table(columns=cols, data=rows)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict):
    config_name = cfg['config_name']
    is_blt      = (config_name == 'C5')
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"  Config: {config_name} | Device: {device}")
    print(f"{'='*50}")

    # --- Load data ---
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_data(
        data_dir    = cfg['data_dir'],
        batch_size  = cfg['batch_size'],
        max_src_len = cfg['max_src_len'],
        max_tgt_len = cfg['max_tgt_len'],
        byte_level  = is_blt,
    )

    # --- Build model ---
    if is_blt:
        model = BLTTransformer(
            d_model        = cfg['d_model'],
            num_heads      = cfg['num_heads'],
            num_layers     = cfg['num_layers'],
            d_ff           = cfg['d_ff'],
            patch_size     = cfg['patch_size'],
            d_local        = cfg.get('d_local', cfg.get('d_byte', 128)),
            local_heads    = cfg['local_heads'],
            dropout        = cfg['dropout'],
        ).to(device)
    else:
        model = build_model(
            config_name    = config_name,
            src_vocab_size = len(src_vocab),
            tgt_vocab_size = len(tgt_vocab),
            pad_idx        = src_vocab.pad_idx,
            d_model        = cfg['d_model'],
            num_heads      = cfg['num_heads'],
            num_layers     = cfg['num_layers'],
            d_ff           = cfg['d_ff'],
            max_len        = cfg['max_len'],
            dropout        = cfg['dropout'],
            num_kv_groups  = cfg['num_kv_groups'],
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # --- Optimizer + scheduler ---
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999),
                      eps=1e-8, weight_decay=1e-2)
    scheduler = NoamScheduler(optimizer, cfg['d_model'], cfg['warmup_steps'])

    # Mixed-precision scaler — only active on CUDA; None = fp32 on CPU
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler(device.type) if use_amp else None
    if use_amp:
        print(f"  AMP enabled (fp16 + GradScaler)")

    # --- WandB ---
    if WANDB_AVAILABLE:
        wandb.init(project='ANLP_A1', name=config_name, config=cfg, reinit=True)

    # --- Training loop ---
    os.makedirs('outputs', exist_ok=True)
    ckpt_path    = f"outputs/{config_name}.pt"
    best_val     = float('inf')
    history      = {'train_loss': [], 'val_loss': []}
    patience     = cfg.get('patience', 5)
    no_improve   = 0

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, cfg, is_blt, scaler=scaler)
        val_loss   = evaluate(model, val_loader, device, is_blt)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if WANDB_AVAILABLE:
            wandb.log({'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss})

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    val_loss,
                'config':      cfg,
                # NOTE: we do NOT save src_vocab/tgt_vocab here — pickling custom
                # classes causes UnpicklingError with newer PyTorch security policy.
                # Vocab is rebuilt cheaply from load_data() at eval time instead.
            }, ckpt_path)
            print(f"  Saved best checkpoint ({val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}!")
                break

    # --- Plot training curves ---
    from utils import plot_training_curves
    plot_training_curves(history, config_name, save_dir='outputs')

    # --- Test evaluation ---
    print("\nEvaluating on test set...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    metrics = run_metrics(model, test_loader, src_vocab, tgt_vocab, device, is_blt)
    print(f"\n--- {config_name} Test Metrics ---")
    for k, v in metrics.items():
        if not k.startswith('_') and isinstance(v, (int, float)):
            print(f"  {k:20s}: {v:.4f}")

    if WANDB_AVAILABLE:
        log_dict = {f'test/{k}': v for k, v in metrics.items() if not k.startswith('_')}
        if '_pred_table' in metrics:
            log_dict['test/predictions'] = metrics.pop('_pred_table')
        wandb.log(log_dict)
        wandb.finish()

    return metrics, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='C1',
                        choices=['C1', 'C2', 'C3', 'C4', 'C5'],
                        help='Which ablation config to train')
    parser.add_argument('--data_dir',   type=str, default=DEFAULT_CONFIG['data_dir'])
    parser.add_argument('--epochs',     type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **vars(args)}
    train(cfg)
