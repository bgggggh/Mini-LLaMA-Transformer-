import argparse
import numpy as np
from cs336_basics.tokenizer import Tokenizer, train_bpe_chunked
import pickle

def tokenize_file_chunked(tokenizer, input_path, output_path, chunk_size=1024*1024):
    """divide tokenize large file"""
    all_ids = []
    total = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if buffer:
                    ids = tokenizer.encode(buffer)
                    all_ids.extend(ids)
                    total += len(ids)
                break
            
            buffer += chunk
            last_newline = buffer.rfind('\n')
            if last_newline == -1:
                continue
            
            to_process = buffer[:last_newline + 1]
            buffer = buffer[last_newline + 1:]
            
            ids = tokenizer.encode(to_process)
            all_ids.extend(ids)
            total += len(ids)
            
            if total % 1000000 < len(ids):
                print(f"    {total:,} tokens processed...")
    
    arr = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, arr)
    print(f"  Saved {len(arr):,} tokens to {output_path}")

def main(args):
    print("Step 1: Training BPE tokenizer...")
    vocab, merges = train_bpe_chunked(
        args.input_path, args.vocab_size, ["<|endoftext|>"]
    )
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    #save tokenizer
    with open("data/tokenizer.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print("  Saved tokenizer to data/tokenizer.pkl")
    
    print("Step 2: Tokenizing training data...")
    tokenize_file_chunked(tokenizer, args.input_path, args.output_path)
    
    if args.val_input_path:
        print("Step 3: Tokenizing validation data...")
        tokenize_file_chunked(tokenizer, args.val_input_path, args.val_output_path)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--output_path", default="data/train_tokens.npy")
    parser.add_argument("--val_input_path", default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--val_output_path", default="data/val_tokens.npy")
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()
    main(args)