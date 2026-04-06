import argparse
import pickle
import torch
from cs336_basics.model import TransformerLM, AdamW, load_checkpoint, softmax
from cs336_basics.tokenizer import Tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device='cpu'):
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids)
            next_logits = logits[0, -1, :] / temperature
            probs = softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
    
    return tokenizer.decode(ids[0].tolist())

def main(args):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("Loading tokenizer...")
    with open(args.tokenizer_path, "rb") as f:
        data = pickle.load(f)
    tokenizer = Tokenizer(data["vocab"], data["merges"], ["<|endoftext|>"])
    
    print("Loading model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)
    
    optimizer = AdamW(model.parameters())
    load_checkpoint(args.checkpoint, model, optimizer)
    
    print(f"\nPrompt: {args.prompt}\n")
    text = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, device)
    print(f"Generated:\n{text}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/checkpoint_final.pt")
    p.add_argument("--tokenizer_path", default="data/tokenizer.pkl")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()
    main(args)