"""
dataset.py — Data Loading: Tokenized (C1–C4) + Byte-Level (C5)
================================================================

TWO MODES
---------
1. TOKENIZED (C1–C4): Character-level vocabulary over the cipher binary string
   and the plaintext. Returns token IDs.

2. BYTE-LEVEL (C5 / BLT): No vocabulary. Raw UTF-8 bytes (0–255) are returned
   directly. This is what "token-free" means — no tokenization step at all.

TOKENIZED FORMAT (Teacher Forcing)
------------------------------------
    encoder input:  [cipher token IDs]
    decoder input:  [<bos>, t1, t2, ..., tN]  ← shifted target
    decoder target: [t1, t2, ..., tN, <eos>]  ← what we predict at each step

BYTE-LEVEL FORMAT
------------------
    src_bytes: raw cipher bytes  (each '0'/'1' char → its ASCII value)
    tgt_bytes: raw plaintext bytes (UTF-8 encoded)
    Loss is cross-entropy over 256 possible byte values at each position.
"""

import os
import re
import random
from typing import Tuple, List, Dict
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ===========================================================================
# Tokenized Mode (C1 – C4)
# ===========================================================================

class Vocabulary:
    """
    Character-level vocabulary with 4 special tokens.

    Special tokens:
        <pad> = 0  — padding (masked in loss)
        <bos> = 1  — beginning of sequence (decoder start)
        <eos> = 2  — end of sequence (decoder stop signal)
        <unk> = 3  — unknown character (fallback for val/test chars)
    """

    PAD, BOS, EOS, UNK = '<pad>', '<bos>', '<eos>', '<unk>'

    def __init__(self):
        self.token2idx: Dict[str, int] = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}

    def build_from_texts(self, texts: List[str]) -> None:
        for ch in sorted(set(ch for t in texts for ch in t)):
            if ch not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[ch] = idx
                self.idx2token[idx] = ch

    def encode(self, text: str) -> List[int]:
        return [self.token2idx.get(ch, self.token2idx[self.UNK]) for ch in text]

    def decode(self, ids: List[int]) -> str:
        skip = {self.token2idx[s] for s in (self.PAD, self.BOS, self.EOS, self.UNK)}
        return ''.join(self.idx2token[i] for i in ids if i not in skip)

    def __len__(self):
        return len(self.token2idx)

    @property
    def pad_idx(self): return self.token2idx[self.PAD]
    @property
    def bos_idx(self): return self.token2idx[self.BOS]
    @property
    def eos_idx(self): return self.token2idx[self.EOS]


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer from scratch to satisfy TA requirements.
    Merges most frequent adjacent pairs of tokens into a single new token.
    """
    PAD, BOS, EOS, UNK = '<pad>', '<bos>', '<eos>', '<unk>'

    def __init__(self, vocab_size=500, is_ciphertext: bool = False):
        self.vocab_size = vocab_size
        self.is_ciphertext = is_ciphertext  # if True: pre-chunk binary into 8-bit bytes
        self.token2idx: Dict[str, int] = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}
        self.merges: Dict[Tuple[int, int], int] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into base tokens.

        For ciphertext (binary strings): chunk every 8 bits into a byte token
        '<0xHH>' — gives 256 base tokens instead of just {'0', '1'}.
        Sequences become ~8× shorter, and BPE learns real byte-level patterns.

        For plaintext: standard character-level split.
        """
        if self.is_ciphertext:
            bits = re.sub(r'[^01]', '', text)           # strip non-binary chars
            remainder = len(bits) % 8
            if remainder:                                # pad to multiple of 8
                bits += '0' * (8 - remainder)
            return [f'<0x{int(bits[i:i+8], 2):02X}>'   # e.g. '01000001' → '<0x41>'
                    for i in range(0, len(bits), 8)]
        return list(text)                               # plaintext: char-level

    def build_from_texts(self, texts: List[str]) -> None:
        # Base vocabulary from unique tokens (byte-chunks for cipher, chars for plain)
        all_base = sorted(set(tok for t in texts for tok in self._tokenize(t)))
        idx = 4
        for tok in all_base:
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok
            idx += 1

        # Convert texts to lists of initial token IDs using _tokenize
        dataset_ids = [
            [self.token2idx.get(tok, self.token2idx[self.UNK]) for tok in self._tokenize(text)]
            for text in texts
        ]

        # Learn BPE merges
        num_merges = self.vocab_size - idx
        for _ in range(num_merges):
            stats = {}
            for ids in dataset_ids:
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    stats[pair] = stats.get(pair, 0) + 1
            
            if not stats:
                break
                
            best_pair = max(stats, key=stats.get)
            new_idx = idx
            idx += 1
            
            self.merges[best_pair] = new_idx
            self.idx2token[new_idx] = self.idx2token[best_pair[0]] + self.idx2token[best_pair[1]]
            
            # Apply merge to the dataset
            new_dataset_ids = []
            for ids in dataset_ids:
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                        new_ids.append(new_idx)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                new_dataset_ids.append(new_ids)
            dataset_ids = new_dataset_ids

    def encode(self, text: str) -> List[int]:
        # Pre-tokenize with _tokenize (handles 8-bit chunking for ciphertext)
        ids = [self.token2idx.get(tok, self.token2idx[self.UNK]) for tok in self._tokenize(text)]
        
        while len(ids) >= 2:
            # Find the pair that was merged earliest (lowest new_idx)
            stats = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                if pair in self.merges:
                    stats[pair] = self.merges[pair]
                    
            if not stats:
                break
                
            # Get the pair with the smallest merge index (learned earliest)
            best_pair = min(stats, key=stats.get)
            new_idx = self.merges[best_pair]
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        return ids

    def decode(self, ids: List[int]) -> str:
        skip = {self.token2idx[s] for s in (self.PAD, self.BOS, self.EOS, self.UNK)}
        return ''.join(self.idx2token.get(i, '') for i in ids if i not in skip)

    def __len__(self):
        return len(self.idx2token)

    @property
    def pad_idx(self): return self.token2idx[self.PAD]
    @property
    def bos_idx(self): return self.token2idx[self.BOS]
    @property
    def eos_idx(self): return self.token2idx[self.EOS]


class CipherPlainDataset(Dataset):
    """
    Tokenized dataset for C1–C4.
    Returns (src_ids, tgt_ids_with_bos_eos) per sample.
    """

    def __init__(self, cipher_texts, plain_texts, src_vocab, tgt_vocab,
                 max_src_len=512, max_tgt_len=256):
        assert len(cipher_texts) == len(plain_texts)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.samples = []
        for c, p in zip(cipher_texts, plain_texts):
            src_ids = src_vocab.encode(c.strip())[:max_src_len]
            tgt_ids = ([tgt_vocab.bos_idx]
                       + tgt_vocab.encode(p.strip())[:max_tgt_len - 2]
                       + [tgt_vocab.eos_idx])
            self.samples.append((src_ids, tgt_ids))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s, t = self.samples[idx]
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)


def _collate_tokenized(batch, src_pad, tgt_pad):
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=src_pad),
        pad_sequence(tgts, batch_first=True, padding_value=tgt_pad),
    )


# ===========================================================================
# Byte-Level Mode (C5 — BLT)
# ===========================================================================
class ByteDataset(Dataset):
    """
    Token-free dataset for C5 (BLT).
    Converts binary text to raw bytes (0-255).
    PAD byte = 256, BOS byte = 257, EOS byte = 258.
    """
    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258

    def __init__(self, cipher_texts, plain_texts, max_src_bytes=512, max_tgt_bytes=256):
        self.samples = []
        for c, p in zip(cipher_texts, plain_texts):
            c_clean = c.strip()
            if len(c_clean) % 8 != 0:
                c_clean += '0' * (8 - len(c_clean) % 8)
            src_bytes = [int(c_clean[i:i+8], 2) for i in range(0, len(c_clean), 8)][:max_src_bytes]
            tgt_bytes = list(p.strip().encode('utf-8'))[:max_tgt_bytes]
            
            # Append EOS so the model can learn to stop
            tgt_bytes.append(self.EOS_ID)
            self.samples.append((src_bytes, tgt_bytes))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s_bytes, t_bytes = self.samples[idx]
        return torch.tensor(s_bytes, dtype=torch.long), torch.tensor(t_bytes, dtype=torch.long)

def _collate_bytes(batch):
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=256),
        pad_sequence(tgts, batch_first=True, padding_value=256)
    )


# ===========================================================================
# Main loader
# ===========================================================================

def load_data(
    data_dir:    str,
    batch_size:  int   = 32,
    train_frac:  float = 0.8,
    val_frac:    float = 0.1,
    max_src_len: int   = 512,
    max_tgt_len: int   = 256,
    seed:        int   = 42,
    byte_level:  bool  = False,   # True → C5 (BLT), False → C1-C4
):
    """
    Load the Brown cipher dataset and return DataLoaders.

    Args:
        data_dir:   Folder with brown_cipher.txt and brown_plain.txt.
        batch_size: Training batch size.
        train_frac: Fraction for training split.
        val_frac:   Fraction for validation split (rest → test).
        max_src_len: Truncate source sequences to this length.
        max_tgt_len: Truncate target sequences to this length.
        seed:       Random seed for reproducibility.
        byte_level: If True, use raw bytes (C5). If False, use char vocab (C1-C4).

    Returns:
        (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
        For byte_level=True, src_vocab and tgt_vocab are None.
    """
    cipher_path = os.path.join(data_dir, 'brown_cipher.txt')
    plain_path  = os.path.join(data_dir, 'brown_plain.txt')

    with open(cipher_path, 'r', encoding='utf-8') as f:
        cipher_lines = f.readlines()
    with open(plain_path, 'r', encoding='utf-8') as f:
        plain_lines  = f.readlines()

    assert len(cipher_lines) == len(plain_lines)

    # Reproducible shuffle
    pairs = list(zip(cipher_lines, plain_lines))
    random.seed(seed)
    random.shuffle(pairs)
    cipher_lines, plain_lines = zip(*pairs)

    n       = len(cipher_lines)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    splits = {
        'train': (cipher_lines[:n_train],              plain_lines[:n_train]),
        'val':   (cipher_lines[n_train:n_train+n_val], plain_lines[n_train:n_train+n_val]),
        'test':  (cipher_lines[n_train+n_val:],        plain_lines[n_train+n_val:]),
    }

    if byte_level:
        # C5: raw bytes, fixed patches
        datasets = {
            k: ByteDataset(c, p, max_src_len, max_tgt_len)
            for k, (c, p) in splits.items()
        }
        train_loader = DataLoader(datasets['train'], batch_size=batch_size,  shuffle=True,  collate_fn=_collate_bytes, num_workers=0)
        val_loader   = DataLoader(datasets['val'],   batch_size=batch_size,  shuffle=False, collate_fn=_collate_bytes, num_workers=0)
        test_loader  = DataLoader(datasets['test'],  batch_size=1,           shuffle=False, collate_fn=_collate_bytes, num_workers=0)
        print(f"[BLT] Dataset: {len(datasets['train'])} train | {len(datasets['val'])} val | {len(datasets['test'])} test (raw bytes)")
        return train_loader, val_loader, test_loader, None, None

    else:
        # C1–C4: BPE for cipher, character-level for plaintext
        # is_ciphertext=True → 8-bit byte chunking for binary cipher strings.
        # vocab_size=2000 gives BPE enough merges to learn meaningful byte patterns.
        src_vocab = BPETokenizer(vocab_size=2000, is_ciphertext=True)
        tgt_vocab = BPETokenizer(vocab_size=2000, is_ciphertext=False)
        src_vocab.build_from_texts(splits['train'][0])
        tgt_vocab.build_from_texts(splits['train'][1])

        datasets = {
            k: CipherPlainDataset(c, p, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
            for k, (c, p) in splits.items()
        }
        _collate = partial(_collate_tokenized, src_pad=src_vocab.pad_idx, tgt_pad=tgt_vocab.pad_idx)

        train_loader = DataLoader(datasets['train'], batch_size=batch_size,  shuffle=True,  collate_fn=_collate, num_workers=0)
        val_loader   = DataLoader(datasets['val'],   batch_size=batch_size,  shuffle=False, collate_fn=_collate, num_workers=0)
        test_loader  = DataLoader(datasets['test'],  batch_size=1,           shuffle=False, collate_fn=_collate, num_workers=0)

        print(f"Dataset: {len(datasets['train'])} train | {len(datasets['val'])} val | {len(datasets['test'])} test")
        print(f"Src vocab: {len(src_vocab)} | Tgt vocab: {len(tgt_vocab)}")
        return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
