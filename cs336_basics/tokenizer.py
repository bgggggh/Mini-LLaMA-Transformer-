import regex as re
from collections import Counter

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX = re.compile(GPT2_PAT)

def pretokenize(text: str) -> list[bytes]:
    return [match.encode("utf-8") for match in GPT2_REGEX.findall(text)]

def get_pair_counts(word_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    counts = {}
    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts

def merge_pair(word, pair, new_id):
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


    

def pretokenize_with_special_tokens(text: str, special_tokens: list[str]) -> list[bytes]:
    """先按 special tokens 切分文本，再对每段分别 pre-tokenize"""
    if not special_tokens:
        return pretokenize(text)
    
    # 构建匹配 special tokens 的正则
    # 用 re.escape 转义特殊字符，用 | 连接
    # 按长度降序排列，确保长的 token 优先匹配
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_tokens)
    
    # split 会保留捕获组中的分隔符
    chunks = re.split(f"({pattern})", text)
    
    result = []
    special_set = set(special_tokens)
    for chunk in chunks:
        if chunk in special_set:
            pass  # special token 不参与 BPE 训练
        elif chunk:
            result.extend(pretokenize(chunk))
    return result




def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    num_merges = vocab_size - 256 - len(special_tokens)
    
    with open(input_path, "r") as f:
        text = f.read()
    
    tokens = pretokenize_with_special_tokens(text, special_tokens)
    word_counts = {tuple(word): count for word, count in Counter(tokens).items()}
    
    merges = []
    for i in range(num_merges):
        pair_counts = get_pair_counts(word_counts)
        if not pair_counts:
            break
        # best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        new_id = 256 + i

        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = merge_pair(word, best_pair, new_id)
            new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count
        word_counts = new_word_counts

    for i, token in enumerate(special_tokens):
        vocab[256 + num_merges + i] = token.encode("utf-8")
    
    return vocab, merges


class Tokenizer:
    def __init__ (self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    def _apply_merges(self, token_list):
        while len(token_list) >= 2:
            best_pair = None
            best_rank = float('inf')
            for i in range(len(token_list) - 1):
                pair = (token_list[i], token_list[i+1])
                if pair in self.merge_ranks and self.merge_ranks[pair] < best_rank:
                    best_pair = pair
                    best_rank = self.merge_ranks[pair]

            if best_pair == None:
                break

            new_token = best_pair[0] + best_pair[1]
            new_list = []

            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and token_list[i] == best_pair[0] and token_list[i+1] == best_pair[1]:
                    new_list.append(new_token)
                    i += 2
                else:
                    new_list.append(token_list[i])
                    i += 1
            token_list = new_list
        return token_list

    def encode_iterable(self, iterable):
        #yield token id one by one
        for line in iterable:
            for id in self.encode(line):
                yield id

    def encode(self, text: str) -> list[int]:
        ids = []
        
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(t) for t in sorted_tokens)
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]
        
        special_set = set(self.special_tokens)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:
                ids.append(self.vocab_inv[chunk.encode("utf-8")])
            else:
                for pretoken in pretokenize(chunk):
                    token_list = [bytes([b]) for b in pretoken]
                    token_list = self._apply_merges(token_list)
                    ids.extend(self.vocab_inv[t] for t in token_list)
        
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")


from collections import Counter
import os

def train_bpe_chunked(input_path, vocab_size, special_tokens, chunk_size=1024*1024):
    """train BPE by chunks, avoid RAM limitation"""
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    
    print("  Counting word frequencies...")
    word_counts_bytes = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if buffer:
                    tokens = pretokenize_with_special_tokens(buffer, special_tokens)
                    word_counts_bytes.update(tokens)
                break
            
            buffer += chunk
            last_newline = buffer.rfind('\n')
            if last_newline == -1:
                continue
            
            to_process = buffer[:last_newline + 1]
            buffer = buffer[last_newline + 1:]
            
            tokens = pretokenize_with_special_tokens(to_process, special_tokens)
            word_counts_bytes.update(tokens)
    
    print(f"  Unique pre-tokens: {len(word_counts_bytes):,}")
    
    word_counts = {tuple(word): count for word, count in word_counts_bytes.items()}
    del word_counts_bytes  
    
    print("  Running merges...")
    merges = []
    for i in range(num_merges):
        pair_counts = get_pair_counts(word_counts)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        new_id = 256 + i
        
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = merge_pair(word, best_pair, new_id)
            new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count
        word_counts = new_word_counts
        
        if (i + 1) % 100 == 0:
            print(f"    Merge {i+1}/{num_merges}")
    
    for i, token in enumerate(special_tokens):
        vocab[256 + num_merges + i] = token.encode("utf-8")
    
    return vocab, merges