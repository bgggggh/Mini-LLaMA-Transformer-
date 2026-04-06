import argparse
import math
import numpy as np
import torch
from cs336_basics.model import (
    TransformerLM, AdamW, cross_entropy, get_batch,
    get_lr_cosine_schedule, gradient_clipping,
    save_checkpoint, load_checkpoint
)

def evaluate(model, val_data, batch_size, context_length, device, vocab_size, num_batches=10):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(val_data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / num_batches
    return avg_loss, math.exp(avg_loss)

def train(args):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data else None
    print(f"Train tokens: {len(train_data):,}")
    
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps
        )
        for group in optimizer.param_groups:
            group['lr'] = lr
        
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        if step % args.log_every == 0:
            print(f"Step {step:>5d} | Loss {loss.item():.4f} | LR {lr:.6f}")
        
        if val_data is not None and step > 0 and step % args.eval_every == 0:
            val_loss, ppl = evaluate(
                model, val_data, args.batch_size, args.context_length,
                device, args.vocab_size
            )
            print(f"         | Val Loss {val_loss:.4f} | Perplexity {ppl:.2f}")
        
        if step > 0 and step % args.save_every == 0:
            path = f"{args.checkpoint_dir}/checkpoint_{step}.pt"
            save_checkpoint(model, optimizer, step, path)
            print(f"         | Saved checkpoint to {path}")
    
    save_checkpoint(model, optimizer, args.max_steps, f"{args.checkpoint_dir}/checkpoint_final.pt")
    print("Training complete!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="data/train_tokens.npy")
    p.add_argument("--val_data", default="data/val_tokens.npy")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = p.parse_args()
    
    import os
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)