import torch
import math
from einops import rearrange
import numpy as np

def linear(weights, in_features):
    return in_features @ weights.T
    
def embedding(weights, token_ids):
    return weights[token_ids]

def silu(x):
    return x * torch.sigmoid(x)

def swiglu(w1, w2, w3, x):
    return linear(w2, silu(linear(w1, x)) * linear(w3, x))

def softmax(x, dim):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def rmsnorm(x, weights, eps=1e-5):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weights

def scaled_dot_product_attention(Q, K ,V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = softmax(scores, dim=-1)
    return weights @ V

def rope(x, token_positions, theta=10000.0):
    d_k = x.shape[-1]
    assert d_k % 2 == 0

    i = torch.arange(0, d_k, 2, device=x.device, dtype=x.dtype)
    freqs = 1.0 / (theta ** (i / d_k))
    angles = token_positions.unsqueeze(-1).float() * freqs
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    x_even = x[..., 0::2] 
    x_odd  = x[..., 1::2]

    out_even = x_even * cos_angles - x_odd * sin_angles
    out_odd  = x_even * sin_angles + x_odd * cos_angles

    out = torch.stack([out_even, out_odd], dim=-1)  
    return out.reshape(x.shape)


def multihead_self_attention(in_features, q_proj, k_proj, v_proj, o_proj, num_heads, rope_fn=None, token_positions=None):
    Q = linear(q_proj, in_features)
    K = linear(k_proj, in_features)
    V = linear(v_proj, in_features)

    Q = rearrange(Q, '... s (h d) -> ... h s d', h=num_heads)
    K = rearrange(K, '... s (h d) -> ... h s d', h=num_heads)
    V = rearrange(V, '... s (h d) -> ... h s d', h=num_heads)

    if rope_fn is not None:
        Q = rope_fn(Q, token_positions)
        K = rope_fn(K, token_positions)

    seq_len = Q.shape[-2]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))

    attn_output = scaled_dot_product_attention(Q, K, V, mask)
    attn_output = rearrange(attn_output, '... h s d -> ... s (h d)')

    return linear(o_proj, attn_output)

def transformer_block(x, weights, num_heads, theta):
    #Attention
    residual = x
    x = rmsnorm(x, weights['ln1.weight'])
    x = multihead_self_attention(
        x,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.output_proj.weight'],
        num_heads,
        rope_fn=lambda q, pos: rope(q, pos, theta),
        token_positions=torch.arange(x.shape[-2], device=x.device).unsqueeze(0)
    )
    x = x + residual
    
    # FFN
    residual = x
    x = rmsnorm(x, weights['ln2.weight'])
    x = swiglu(weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'], x)
    x = x + residual
    
    return x

def transformer_lm(x_ids, weights, num_layers, num_heads, theta):
    # Embedding
    x = embedding(weights['token_embeddings.weight'], x_ids)
    
    # N layers
    for i in range(num_layers):
        layer_weights = {k.replace(f'layers.{i}.', ''): v 
                        for k, v in weights.items() if k.startswith(f'layers.{i}.')}
        x = transformer_block(x, layer_weights, num_heads, theta)
    
    # Final norm + LM head
    x = rmsnorm(x, weights['ln_final.weight'])
    x = linear(weights['lm_head.weight'], x)
    return x


def cross_entropy(inputs, targets):
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    loss = -log_probs[torch.arange(len(targets)), targets]
    return loss.mean()


def get_batch(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.stack([torch.from_numpy(dataset[s:s+context_length].copy()) for s in starts]).long()
    y = torch.stack([torch.from_numpy(dataset[s+1:s+1+context_length].copy()) for s in starts]).long()
    return x.to(device), y.to(device)

def gradient_clipping(parameters, max_l2_norm):
    parameters = list(parameters)
    total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in parameters if p.grad is not None))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                t = state['step']
                m, v = state['m'], state['v']
                
                m.mul_(b1).add_(p.grad, alpha=1-b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1-b2)
                
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                
                p.data.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                p.data.add_(p.data, alpha=-lr * wd)

def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)
    elif it >= cosine_cycle_iters:
        return min_lr
    else:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': iteration}, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']



import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
    
    def forward(self, x):
        return swiglu(self.w1, self.w2, self.w3, x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.theta = theta
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.output_proj = nn.Parameter(torch.empty(d_model, d_model))
        for w in [self.q_proj, self.k_proj, self.v_proj, self.output_proj]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    
    def forward(self, x):
        rope_fn = lambda q, pos: rope(q, pos, self.theta)
        positions = torch.arange(x.shape[-2], device=x.device).unsqueeze(0)
        return multihead_self_attention(
            x, self.q_proj, self.k_proj, self.v_proj, self.output_proj,
            self.num_heads, rope_fn, positions
        )

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, theta=10000.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.kaiming_uniform_(self.lm_head, a=math.sqrt(5))
    
    def forward(self, x_ids):
        x = self.token_embeddings(x_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return linear(self.lm_head, x)