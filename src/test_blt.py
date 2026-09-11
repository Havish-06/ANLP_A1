import torch
import torch.nn as nn
from models.blt import BLTTransformer
from train import byte_loss

def test_blt():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Instantiate the BLTTransformer using the new Conv1d code
    model = BLTTransformer(
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        patch_size=4,
        dropout=0.1,
        max_len=64
    ).to(device)
    
    print("Model initialized. Patch size:", model.patch_size)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    
    B, L = 2, 32
    src = torch.randint(0, 256, (B, L)).to(device)
    
    # 2. Simulate training step with shifted inputs
    tgt = torch.randint(0, 256, (B, L)).to(device)
    
    bos_chunk = torch.full((B, model.patch_size), 256, dtype=torch.long, device=device)
    bos_chunk[:, 0] = 257
    tgt_input = torch.cat([bos_chunk, tgt[:, :-model.patch_size]], dim=1)
    
    print("\n[Train Step]")
    print("Input shape:", src.shape)
    print("Tgt Input shape:", tgt_input.shape)
    
    logits = model(src, tgt_input)
    print("Logits shape:", logits.shape)
    
    loss = byte_loss(logits, tgt)
    print("Loss calculated:", loss.item())
    
    loss.backward()
    print("Backward pass successful!")
    
    # 3. Simulate greedy decode
    print("\n[Inference Step]")
    # In reality greedy_decode uses max_len for the generated sequence length
    decoded = model.greedy_decode(src, max_len=L)
    print("Decoded shape:", decoded.shape)
    print("Test passed!")

if __name__ == "__main__":
    test_blt()
